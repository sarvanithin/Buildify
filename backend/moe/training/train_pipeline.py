"""
Improved MOE Training Pipeline — uses HouseGAN pseudo-labels for better spatial reasoning.

Loss functions:
  1. Room sizing MSE         — match predicted dimensions to HouseGAN/IRC targets
  2. IRC compliance loss     — penalise rooms smaller than code minimums
  3. Adjacency reward loss   — reward correct room adjacency from bubble diagram
  4. Load balancing loss     — prevent expert collapse (all samples use same expert)
  5. Footprint coverage loss — total room area should match target sqft

Usage:
  python -m moe.training.train_pipeline \
    --dataset moe/data/train.pt \
    --val-dataset moe/data/val.pt \
    --out moe/weights/buildify_moe.pt \
    --epochs 200
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from moe.config import MOEConfig
from moe.model import BuildifyMOE
from moe.data import IRC_ROOM_SPECS, ADJACENCY_RULES, encode_constraints
from moe.training.generate_dataset import BuildifyDataset, generate_dataset


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate: constraint vecs can be stacked, rooms are variable-length.
    Returns a dict with tensors for loss computation.
    """
    constraints = torch.stack([b["constraints"] for b in batch])
    rooms       = [b["rooms"] for b in batch]
    footprints  = [b["footprint"] for b in batch]
    adj_pairs   = [b["adj_pairs"] for b in batch]
    return {
        "constraints": constraints,
        "rooms":       rooms,
        "footprints":  footprints,
        "adj_pairs":   adj_pairs,
    }


# ── Loss functions ────────────────────────────────────────────────────────────

def room_sizing_loss(
    expert_weights: torch.Tensor,  # (B, num_experts)
    batch_rooms: List[List[Dict]],
    batch_constraints: List[dict],
    config: MOEConfig,
) -> torch.Tensor:
    """
    MSE loss on predicted room dimensions vs HouseGAN/IRC target dimensions.

    Expert weights modulate sizing — we supervise by asking: given the target
    room sizes from HouseGAN, which expert combination minimises size error?

    Simplified: treat this as a regression target on the sizing factor t ∈ [0,1]
    which interpolates from IRC min → IRC standard → premium.
    """
    losses = []
    for i, rooms in enumerate(batch_rooms):
        sqft = batch_constraints[i].get("sqft", 1800) if batch_constraints else 1800
        sf_target = min(1.0, max(0.0, (sqft - 1200) / 2800))

        for room in rooms:
            rtype = room["type"]
            specs = IRC_ROOM_SPECS.get(rtype)
            if not specs:
                continue

            min_w, min_h, std_w, std_h, prem_w, prem_h = specs
            target_w = float(room["width"])
            target_h = float(room["height"])

            # What t value would produce these dimensions?
            if std_w > min_w:
                t_w = (target_w - min_w) / (prem_w - min_w + 1e-6)
            else:
                t_w = 0.5
            t_w = max(0.0, min(1.0, t_w))

            # Expert weights influence sizing — model predicts t_w via sizing expert
            # Expert 0 = "Room Sizing" expert (from EXPERT_NAMES)
            sizing_weight = expert_weights[i, 0]
            predicted_t = torch.sigmoid(sizing_weight * 4 - 2)  # map to 0-1

            losses.append(F.mse_loss(predicted_t, torch.tensor(t_w, device=expert_weights.device)))

    if not losses:
        return torch.tensor(0.0, device=expert_weights.device)
    return torch.stack(losses).mean()


def irc_compliance_loss(
    expert_weights: torch.Tensor,
    batch_rooms: List[List[Dict]],
    config: MOEConfig,
) -> torch.Tensor:
    """
    Penalise expert weight configurations that would produce code-violating rooms.
    If predicted room sizes (from expert weights) are below IRC minimums, apply loss.
    """
    device = expert_weights.device
    losses = []

    for i, rooms in enumerate(batch_rooms):
        for room in rooms:
            rtype = room["type"]
            specs = IRC_ROOM_SPECS.get(rtype)
            if not specs:
                continue

            min_w, min_h = specs[0], specs[1]
            actual_w = float(room["width"])
            actual_h = float(room["height"])

            # Compliance ratio: how close are we to minimum?
            w_ratio = actual_w / min_w
            h_ratio = actual_h / min_h

            if w_ratio < 1.0 or h_ratio < 1.0:
                # Penalise: small expert weights should have been avoided
                # Cost expert (index 1) should not dominate for undersized rooms
                cost_weight = expert_weights[i, 1]
                violation = max(0.0, 1.0 - min(w_ratio, h_ratio))
                losses.append(cost_weight * violation)

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def adjacency_reward_loss(
    expert_weights: torch.Tensor,
    batch_rooms: List[List[Dict]],
    batch_adj_pairs: List[List],
    config: MOEConfig,
) -> torch.Tensor:
    """
    Reward when spatially placed rooms respect required adjacency.
    For rooms with high adjacency requirement (>0.8), compute distance penalty
    if rooms are placed far apart.
    """
    device = expert_weights.device
    losses = []

    for i, (rooms, adj_pairs) in enumerate(zip(batch_rooms, batch_adj_pairs)):
        room_by_idx = {j: r for j, r in enumerate(rooms)}

        for pair in adj_pairs:
            if len(pair) != 3:
                continue
            idx_a, idx_b, strength = pair
            if strength < 0.8:
                continue

            r_a = room_by_idx.get(idx_a)
            r_b = room_by_idx.get(idx_b)
            if r_a is None or r_b is None:
                continue

            # Centre-to-centre distance
            cx_a = r_a["x"] + r_a["width"]  / 2
            cy_a = r_a["y"] + r_a["height"] / 2
            cx_b = r_b["x"] + r_b["width"]  / 2
            cy_b = r_b["y"] + r_b["height"] / 2
            dist = math.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)

            # Expected adjacency: rooms should be within ~1.5× their average size
            avg_size = (r_a["width"] + r_a["height"] + r_b["width"] + r_b["height"]) / 4
            threshold = avg_size * 2.5

            if dist > threshold:
                # Adjacency expert (index 2) should compensate — if it's weak, penalise
                adj_weight = expert_weights[i, 2]
                excess = (dist - threshold) / threshold
                losses.append((1.0 - adj_weight) * min(excess, 2.0))

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def footprint_coverage_loss(
    expert_weights: torch.Tensor,
    batch_rooms: List[List[Dict]],
    batch_footprints: List,
    config: MOEConfig,
) -> torch.Tensor:
    """
    Total room area should be ~82% of footprint (wall + mechanical efficiency).
    """
    device = expert_weights.device
    losses = []

    for i, (rooms, fp) in enumerate(zip(batch_rooms, batch_footprints)):
        W, H = fp
        footprint_area = W * H
        room_area = sum(r["width"] * r["height"] for r in rooms
                        if r.get("zone") != "outdoor")
        target_area = footprint_area * 0.82

        ratio = room_area / max(target_area, 1)
        coverage_error = abs(1.0 - ratio)
        losses.append(torch.tensor(coverage_error, device=device))

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def load_balance_loss(gate_scores: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary loss to prevent all tokens routing to same experts.
    Maximises entropy of expert usage across the batch.
    """
    # gate_scores: (B, num_experts)
    avg = gate_scores.mean(dim=0)  # (num_experts,) — average usage per expert
    # Ideal: uniform = 1/num_experts for each expert
    target = torch.ones_like(avg) / avg.size(0)
    return F.mse_loss(avg, target)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(
    model: BuildifyMOE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: MOEConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    metrics = {
        "loss": 0, "sizing": 0, "irc": 0, "adj": 0, "coverage": 0, "balance": 0
    }
    n_batches = 0

    for batch in loader:
        constraints = batch["constraints"].to(device)
        rooms       = batch["rooms"]
        footprints  = batch["footprints"]
        adj_pairs   = batch["adj_pairs"]

        # Build raw constraints dicts from encoded vectors (for loss fns that need sqft)
        # We embed sqft info into the first element of constraint vector (see encode_constraints)
        batch_constraints = [{"sqft": int(constraints[i, 0].item() * 5500)}
                              for i in range(len(constraints))]

        optimizer.zero_grad()

        # Forward pass
        with torch.set_grad_enabled(True):
            expert_weights = model.get_expert_weights(constraints)  # (B, num_experts)

            # Load balance loss (always backprop-able)
            l_balance = load_balance_loss(expert_weights)

            # Sizing loss
            l_sizing = room_sizing_loss(expert_weights, rooms, batch_constraints, config)

            # IRC compliance loss
            l_irc = irc_compliance_loss(expert_weights, rooms, config)

            # Adjacency reward loss
            l_adj = adjacency_reward_loss(expert_weights, rooms, adj_pairs, config)

            # Footprint coverage loss
            l_coverage = footprint_coverage_loss(expert_weights, rooms, footprints, config)

            total_loss = (
                l_sizing  * 1.0 +
                l_irc     * config.compliance_weight +
                l_adj     * config.adjacency_weight +
                l_coverage * 0.2 +
                l_balance  * config.load_balance_weight
            )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        metrics["loss"]     += total_loss.item()
        metrics["sizing"]   += l_sizing.item()
        metrics["irc"]      += l_irc.item()
        metrics["adj"]      += l_adj.item()
        metrics["coverage"] += l_coverage.item()
        metrics["balance"]  += l_balance.item()
        n_batches += 1

    for k in metrics:
        metrics[k] /= max(n_batches, 1)

    return metrics


@torch.no_grad()
def val_epoch(
    model: BuildifyMOE,
    loader: DataLoader,
    config: MOEConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    metrics = {"loss": 0, "sizing": 0, "irc": 0}
    n_batches = 0

    for batch in loader:
        constraints = batch["constraints"].to(device)
        rooms       = batch["rooms"]

        expert_weights = model.get_expert_weights(constraints)
        batch_constraints = [{"sqft": int(constraints[i, 0].item() * 5500)}
                              for i in range(len(constraints))]

        l_sizing = room_sizing_loss(expert_weights, rooms, batch_constraints, config)
        l_irc    = irc_compliance_loss(expert_weights, rooms, config)
        l_balance = load_balance_loss(expert_weights)
        total = l_sizing + l_irc * config.compliance_weight + l_balance * 0.1

        metrics["loss"]   += total.item()
        metrics["sizing"] += l_sizing.item()
        metrics["irc"]    += l_irc.item()
        n_batches += 1

    for k in metrics:
        metrics[k] /= max(n_batches, 1)
    return metrics


# ── Main entry ────────────────────────────────────────────────────────────────

def train(
    dataset_path: str = "moe/data/train.pt",
    val_path: Optional[str] = None,
    out_path: str = "moe/weights/buildify_moe.pt",
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    generate_data: bool = False,
    num_samples: int = 10_000,
    use_housegan: bool = True,
    device_str: str = "auto",
) -> BuildifyMOE:
    """Full training run for the Buildify MOE model."""

    # Device selection
    if device_str == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    else:
        device = torch.device(device_str)
    print(f"[Train] Device: {device}")

    config = MOEConfig()
    config.learning_rate = lr
    config.batch_size    = batch_size
    config.epochs        = epochs

    # ── Dataset ──────────────────────────────────────────────────────────────
    if generate_data or not Path(dataset_path).exists():
        print(f"[Train] Generating dataset ({num_samples} samples)...")
        from moe.training.generate_dataset import generate_dataset
        ds = generate_dataset(num_samples, use_housegan, dataset_path)
    else:
        print(f"[Train] Loading dataset from {dataset_path}")
        ds = BuildifyDataset.from_file(dataset_path)

    print(f"[Train] Dataset size: {len(ds)}")

    # Train/val split
    if val_path and Path(val_path).exists():
        train_ds = ds
        val_ds   = BuildifyDataset.from_file(val_path)
    else:
        val_size  = max(500, len(ds) // 10)
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BuildifyMOE(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model params: {n_params:,}")

    # Load existing weights for fine-tuning if available
    weights_path = Path(out_path)
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[Train] Fine-tuning from existing weights: {out_path}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[Train] Starting training — {epochs} epochs")
    print(f"  batch_size={batch_size}  lr={lr}")
    print(f"  train={len(train_ds)}  val={len(val_ds)}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, config, device)
        val_metrics   = val_epoch(model, val_loader, config, device)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val":   val_metrics,
            "lr":    current_lr,
        })

        # Logging
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"(sz={train_metrics['sizing']:.3f} "
                f"irc={train_metrics['irc']:.3f} "
                f"adj={train_metrics['adj']:.3f}) | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"lr={current_lr:.2e} | {elapsed:.1f}s"
            )

        # Checkpoint best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": vars(config),
                "history": history,
            }, out_path)
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"\n[Train] Early stopping at epoch {epoch} "
                      f"(no improvement for {config.early_stop_patience} epochs)")
                break

    print(f"\n[Train] Done. Best val_loss={best_val_loss:.4f} → {out_path}")
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Buildify MOE model")
    parser.add_argument("--dataset",       type=str, default="moe/data/train.pt")
    parser.add_argument("--val-dataset",   type=str, default=None)
    parser.add_argument("--out",           type=str, default="moe/weights/buildify_moe.pt")
    parser.add_argument("--epochs",        type=int, default=200)
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--generate",      action="store_true",
                        help="Generate training data if not found")
    parser.add_argument("--num-samples",   type=int, default=10_000)
    parser.add_argument("--no-housegan",   action="store_true")
    parser.add_argument("--device",        type=str, default="auto")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        val_path=args.val_dataset,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        generate_data=args.generate,
        num_samples=args.num_samples,
        use_housegan=not args.no_housegan,
        device_str=args.device,
    )
