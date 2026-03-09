"""
Training pipeline for the Buildify MOE model.

Multi-objective loss:
  L = L_coords + L_types + α·L_compliance + β·L_adjacency + γ·L_balance

Usage:
  python -m moe.train              # full training
  python -m moe.train --smoke-test # quick 10-epoch test
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import MOEConfig
from .model import BuildifyMOE
from .data import create_dataloaders, IRC_ROOM_SPECS, ADJACENCY_RULES


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class BuildifyLoss(nn.Module):
    """Multi-objective loss for floor plan generation."""

    def __init__(self, config: MOEConfig):
        super().__init__()
        self.config = config
        self.coord_loss = nn.SmoothL1Loss()
        self.type_loss = nn.CrossEntropyLoss()

    def forward(self, predictions: dict, targets: dict, aux_loss: torch.Tensor) -> dict:
        rooms = predictions["rooms"]
        num_rooms = targets["num_rooms"]
        batch_size = num_rooms.shape[0]
        max_r = num_rooms.max().item()

        # 1. Coordinate loss (x, y, w, h)
        pred_coords = rooms["coords"][:, :max_r]
        target_coords = targets["coord_targets"][:, :max_r]
        coord_mask = torch.arange(max_r, device=num_rooms.device).unsqueeze(0) < num_rooms.unsqueeze(1)
        coord_mask = coord_mask.unsqueeze(-1).expand_as(pred_coords)
        L_coords = self.coord_loss(pred_coords * coord_mask, target_coords * coord_mask)

        # 2. Room type classification loss
        pred_types = rooms["type_logits"][:, :max_r]  # (batch, rooms, num_types)
        target_types = targets["type_targets"][:, :max_r]
        pred_flat = pred_types.reshape(-1, self.config.num_room_types)
        target_flat = target_types.reshape(-1)
        L_types = self.type_loss(pred_flat, target_flat)

        # 3. Compliance penalty (rooms that violate IRC minimums)
        L_compliance = self._compliance_loss(pred_coords, rooms["type_logits"][:, :max_r],
                                              coord_mask[:, :, 0])

        # 4. Adjacency reward (rooms that should be close)
        L_adjacency = self._adjacency_loss(pred_coords, rooms["type_logits"][:, :max_r],
                                            coord_mask[:, :, 0])

        # 5. Load balancing (from gating)
        L_balance = aux_loss

        # Weighted total
        total = (L_coords
                 + L_types
                 + self.config.compliance_weight * L_compliance
                 + self.config.adjacency_weight * L_adjacency
                 + self.config.load_balance_weight * L_balance)

        return {
            "total": total,
            "coords": L_coords.item(),
            "types": L_types.item(),
            "compliance": L_compliance.item(),
            "adjacency": L_adjacency.item(),
            "balance": L_balance.item(),
        }

    def _compliance_loss(self, coords, type_logits, mask):
        """Penalize rooms smaller than IRC minimums."""
        # Get predicted room types
        type_indices = type_logits.argmax(dim=-1)  # (batch, rooms)
        widths = coords[:, :, 2]   # normalized widths
        heights = coords[:, :, 3]  # normalized heights

        # Minimum sizes (normalized roughly)
        min_w = 0.1   # ~5ft in a 50ft house
        min_h = 0.1

        violations_w = F.relu(min_w - widths) * mask.float()
        violations_h = F.relu(min_h - heights) * mask.float()

        return (violations_w.sum() + violations_h.sum()) / max(1, mask.sum().item())

    def _adjacency_loss(self, coords, type_logits, mask):
        """Reward correct adjacency pairs being close together."""
        type_probs = F.softmax(type_logits, dim=-1)  # (batch, rooms, types)
        centroids = coords[:, :, :2] + coords[:, :, 2:4] / 2  # (batch, rooms, 2)

        total_loss = torch.tensor(0.0, device=coords.device)
        count = 0

        for type_a, type_b, strength in ADJACENCY_RULES[:8]:  # Top 8 rules
            config = self.config
            if type_a in config.ROOM_TYPES and type_b in config.ROOM_TYPES:
                idx_a = config.ROOM_TYPES.index(type_a)
                idx_b = config.ROOM_TYPES.index(type_b)

                # Soft match (probability of being type A/B)
                prob_a = type_probs[:, :, idx_a]  # (batch, rooms)
                prob_b = type_probs[:, :, idx_b]

                # Pairwise distances between rooms
                diff = centroids.unsqueeze(2) - centroids.unsqueeze(1)  # (batch, r, r, 2)
                dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-6)  # (batch, r, r)

                # Weighted distance (should be small for adjacent rooms)
                weight = prob_a.unsqueeze(2) * prob_b.unsqueeze(1) * strength
                weighted_dist = (dist * weight * mask.unsqueeze(2).float()
                                 * mask.unsqueeze(1).float())

                total_loss = total_loss + weighted_dist.sum()
                count += 1

        return total_loss / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class MOETrainer:
    """Production training loop with early stopping and metrics logging."""

    def __init__(self, config: MOEConfig, device: str = None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available()
                                  else "mps" if torch.backends.mps.is_available()
                                  else "cpu")

        self.model = BuildifyMOE(config).to(self.device)
        self.criterion = BuildifyLoss(config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6,
        )

        self.metrics_log = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Ensure weights directory exists
        weights_dir = Path(__file__).parent / "weights"
        weights_dir.mkdir(exist_ok=True)
        self.save_path = weights_dir / config.model_filename

        print(f"[MOE] Model parameters: {self.model.count_parameters():,}")
        print(f"[MOE] Device: {self.device}")

    def train(self, smoke_test: bool = False):
        """Run full training loop."""
        config = self.config
        if smoke_test:
            config.num_train_samples = 500
            config.num_val_samples = 100
            config.num_test_samples = 100
            config.epochs = 10
            config.batch_size = 32

        print(f"[MOE] Generating training data ({config.num_train_samples} samples)...")
        train_dl, val_dl, test_dl = create_dataloaders(config)
        print(f"[MOE] Training: {len(train_dl)} batches | Val: {len(val_dl)} batches")

        for epoch in range(config.epochs):
            t0 = time.time()

            # Train
            train_metrics = self._train_epoch(train_dl)

            # Validate
            val_metrics = self._validate(val_dl)

            elapsed = time.time() - t0
            lr = self.scheduler.get_last_lr()[0]

            log_entry = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
                "lr": lr,
                "elapsed_s": round(elapsed, 1),
            }
            self.metrics_log.append(log_entry)

            print(
                f"  Epoch {epoch + 1:3d}/{config.epochs} | "
                f"Train: {train_metrics['total']:.4f} | "
                f"Val: {val_metrics['total']:.4f} | "
                f"LR: {lr:.6f} | "
                f"{elapsed:.1f}s"
            )

            # Early stopping
            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self.patience_counter = 0
                self._save_model()
                print(f"  ✓ Saved best model (val_loss={self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= config.early_stop_patience:
                    print(f"  ✗ Early stopping at epoch {epoch + 1}")
                    break

            self.scheduler.step()

        # Test
        print("\n[MOE] Evaluating on test set...")
        self._load_best_model()
        test_metrics = self._validate(test_dl)
        print(f"  Test loss: {test_metrics['total']:.4f}")
        print(f"    Coords: {test_metrics['coords']:.4f}")
        print(f"    Types:  {test_metrics['types']:.4f}")
        print(f"    Compliance: {test_metrics['compliance']:.4f}")
        print(f"    Adjacency:  {test_metrics['adjacency']:.4f}")

        # Save training log
        log_path = Path(__file__).parent / "weights" / "training_log.json"
        log_path.write_text(json.dumps(self.metrics_log, indent=2))
        print(f"\n[MOE] Training log saved to {log_path}")

        return test_metrics

    def _train_epoch(self, dataloader) -> dict:
        self.model.train()
        totals = {"total": 0, "coords": 0, "types": 0,
                  "compliance": 0, "adjacency": 0, "balance": 0}
        n = 0

        for batch in dataloader:
            constraints = batch["constraints"].to(self.device)
            room_tensor = batch["room_tensor"].to(self.device)
            coord_targets = batch["coord_targets"].to(self.device)
            type_targets = batch["type_targets"].to(self.device)
            num_rooms = batch["num_rooms"].to(self.device)

            max_rooms = num_rooms.max().item()

            # Forward
            output = self.model(constraints, max_rooms, teacher_rooms=room_tensor)

            # Loss
            targets = {
                "coord_targets": coord_targets,
                "type_targets": type_targets,
                "num_rooms": num_rooms,
            }
            losses = self.criterion(output, targets, output["aux_loss"])

            # Backward
            self.optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            for k in totals:
                totals[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()
            n += 1

        return {k: round(v / max(n, 1), 4) for k, v in totals.items()}

    @torch.no_grad()
    def _validate(self, dataloader) -> dict:
        self.model.eval()
        totals = {"total": 0, "coords": 0, "types": 0,
                  "compliance": 0, "adjacency": 0, "balance": 0}
        n = 0

        for batch in dataloader:
            constraints = batch["constraints"].to(self.device)
            room_tensor = batch["room_tensor"].to(self.device)
            coord_targets = batch["coord_targets"].to(self.device)
            type_targets = batch["type_targets"].to(self.device)
            num_rooms = batch["num_rooms"].to(self.device)

            max_rooms = num_rooms.max().item()

            output = self.model(constraints, max_rooms)
            targets = {
                "coord_targets": coord_targets,
                "type_targets": type_targets,
                "num_rooms": num_rooms,
            }
            losses = self.criterion(output, targets, output["aux_loss"])

            for k in totals:
                totals[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()
            n += 1

        return {k: round(v / max(n, 1), 4) for k, v in totals.items()}

    def _save_model(self):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }, self.save_path)

    def _load_best_model(self):
        if self.save_path.exists():
            checkpoint = torch.load(self.save_path, map_location=self.device,
                                     weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Buildify MOE")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick 10-epoch test with small data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = MOEConfig()
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size

    trainer = MOETrainer(config)
    trainer.train(smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
