"""
Buildify MOE — Core Mixture of Experts model for floor plan generation.

Architecture:
1. Multi-head self-attention constraint encoder
2. Sparse Top-K gating network (selects top-4 of 8 experts)
3. Residual mixture of expert outputs
4. Autoregressive room decoder (generates rooms sequentially)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import MOEConfig
from .experts import create_experts


# ─────────────────────────────────────────────────────────────────────────────
# Constraint Encoder with Multi-Head Self-Attention
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintEncoder(nn.Module):
    """
    Encodes user constraints (bedrooms, sqft, style, etc.)
    into a dense embedding using multi-head self-attention.
    """

    def __init__(self, config: MOEConfig):
        super().__init__()
        self.proj = nn.Linear(config.input_features, config.embedding_dim)
        self.layer_norm = nn.LayerNorm(config.embedding_dim)

        # Multi-head self-attention over the projected features
        self.attn = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim * 4, config.embedding_dim),
            nn.Dropout(config.dropout),
        )
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_features)
        returns: (batch, embedding_dim)
        """
        # Project to embedding dim
        h = self.layer_norm(self.proj(x))

        # Self-attention (treat as single-token sequence for global context)
        h = h.unsqueeze(1)  # (batch, 1, dim)
        attn_out, _ = self.attn(h, h, h)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ff(h))
        return h.squeeze(1)  # (batch, dim)


# ─────────────────────────────────────────────────────────────────────────────
# Sparse Top-K Gating Network
# ─────────────────────────────────────────────────────────────────────────────

class SoftGating(nn.Module):
    """
    Soft MoE gating: all experts contribute via softmax weights.
    Fully differentiable — no top-k discrete selection, no token dropping.
    Entropy maximisation encourages balanced expert utilisation without
    a separate auxiliary load-balancing loss.
    """

    def __init__(self, config: MOEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.entropy_weight = config.entropy_weight

        self.gate = nn.Sequential(
            nn.Linear(config.embedding_dim, config.gating_hidden_dim),
            nn.GELU(),
            nn.Linear(config.gating_hidden_dim, config.num_experts),
        )

    def forward(self, x: torch.Tensor, training: bool = True):
        """
        x: (batch, embedding_dim)
        returns: (weights, aux_loss)
            weights:  (batch, num_experts) — softmax over all experts
            aux_loss: scalar — entropy regularisation (maximise diversity)
        """
        logits = self.gate(x)                            # (batch, E)
        weights = F.softmax(logits, dim=-1)              # all experts ≥ 0, sum = 1

        # Entropy maximisation: −H encourages equal expert use (no collapse)
        entropy = -(weights * (weights + 1e-8).log()).sum(-1).mean()
        aux_loss = -self.entropy_weight * entropy

        return weights, aux_loss

    def get_all_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Full softmax weights over all experts (for visualisation)."""
        logits = self.gate(x)
        return F.softmax(logits, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Autoregressive Room Decoder
# ─────────────────────────────────────────────────────────────────────────────

class RoomDecoder(nn.Module):
    """
    Autoregressive decoder: generates rooms one at a time,
    conditioned on previously placed rooms to prevent overlaps.

    For each room, outputs: x, y, width, height, room_type logits, zone, is_exterior
    """

    def __init__(self, config: MOEConfig):
        super().__init__()
        self.config = config
        self.max_rooms = config.max_rooms
        self.room_features = config.room_features
        self.output_dim = config.output_dim

        # Room state encoder (encodes already-placed rooms)
        self.room_encoder = nn.GRU(
            input_size=config.room_features,
            hidden_size=config.embedding_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout,
        )

        # Combine context embedding + room state → next room prediction
        self.combiner = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Output heads
        self.coord_head = nn.Linear(config.embedding_dim, 4)      # x, y, w, h
        self.type_head = nn.Linear(config.embedding_dim, config.num_room_types)
        self.zone_head = nn.Linear(config.embedding_dim, 1)       # public(0) vs private(1)
        self.exterior_head = nn.Linear(config.embedding_dim, 1)   # interior(0) vs exterior(1)
        self.stop_head = nn.Linear(config.embedding_dim, 1)       # stop signal

    def forward(self, context: torch.Tensor, num_rooms: int,
                teacher_rooms: torch.Tensor = None) -> dict:
        """
        context: (batch, embedding_dim) — from MOE mixture
        num_rooms: number of rooms to generate
        teacher_rooms: (batch, num_rooms, room_features) — for teacher forcing

        returns dict with keys: coords, type_logits, zones, exteriors, stop_logits
        """
        batch_size = context.shape[0]
        device = context.device

        all_coords = []
        all_type_logits = []
        all_zones = []
        all_exteriors = []
        all_stop_logits = []

        # Initial hidden state for GRU
        h = torch.zeros(2, batch_size, self.config.embedding_dim, device=device)

        # Start token (zeros)
        prev_room = torch.zeros(batch_size, 1, self.room_features, device=device)

        for step in range(num_rooms):
            # Encode room history
            _, h = self.room_encoder(prev_room, h)
            room_state = h[-1]  # last layer hidden (batch, dim)

            # Combine with MOE context
            combined = torch.cat([context, room_state], dim=-1)
            hidden = self.combiner(combined)

            # Generate room parameters
            coords = torch.sigmoid(self.coord_head(hidden))  # (batch, 4) in [0,1]
            type_logits = self.type_head(hidden)               # (batch, num_types)
            zone = torch.sigmoid(self.zone_head(hidden))       # (batch, 1)
            exterior = torch.sigmoid(self.exterior_head(hidden))
            stop_logit = self.stop_head(hidden)

            all_coords.append(coords)
            all_type_logits.append(type_logits)
            all_zones.append(zone)
            all_exteriors.append(exterior)
            all_stop_logits.append(stop_logit)

            # Prepare next step input (teacher forcing or own prediction)
            if teacher_rooms is not None and step < teacher_rooms.shape[1]:
                prev_room = teacher_rooms[:, step:step + 1, :]
            else:
                # Use own prediction
                room_type_idx = type_logits.argmax(dim=-1, keepdim=True).float()
                own_room = torch.cat([
                    coords,
                    room_type_idx / self.config.num_room_types,
                    zone, exterior
                ], dim=-1)
                prev_room = own_room.unsqueeze(1)

        return {
            "coords": torch.stack(all_coords, dim=1),         # (batch, rooms, 4)
            "type_logits": torch.stack(all_type_logits, dim=1),  # (batch, rooms, types)
            "zones": torch.stack(all_zones, dim=1),              # (batch, rooms, 1)
            "exteriors": torch.stack(all_exteriors, dim=1),      # (batch, rooms, 1)
            "stop_logits": torch.stack(all_stop_logits, dim=1),  # (batch, rooms, 1)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main MOE Model
# ─────────────────────────────────────────────────────────────────────────────

class BuildifyMOE(nn.Module):
    """
    Complete Buildify Mixture of Experts model.

    Flow: constraints → encoder → gating → experts → mixture → decoder → rooms
    """

    def __init__(self, config: MOEConfig = None):
        super().__init__()
        self.config = config or MOEConfig()

        self.encoder = ConstraintEncoder(self.config)
        self.gating = SoftGating(self.config)
        self.experts = create_experts(self.config)
        self.decoder = RoomDecoder(self.config)

        # Residual projection for mixture output
        self.mixture_proj = nn.Sequential(
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim),
            nn.LayerNorm(self.config.embedding_dim),
            nn.GELU(),
        )

        # Track expert activations for visualization
        self._last_expert_weights = None

    def forward(self, constraints: torch.Tensor, num_rooms: int,
                teacher_rooms: torch.Tensor = None) -> dict:
        """
        constraints: (batch, input_features) — encoded constraint vector
        num_rooms: how many rooms to generate
        teacher_rooms: optional (batch, num_rooms, room_features) for training

        returns dict with:
            - rooms: decoded room parameters
            - expert_weights: (batch, num_experts) full expert weights
            - aux_loss: load balancing loss
        """
        # 1. Encode constraints
        embedded = self.encoder(constraints)  # (batch, dim)

        # 2. Gate: soft weights over all experts (fully differentiable)
        weights, aux_loss = self.gating(embedded, training=self.training)

        # 3. All experts contribute — weighted sum (no top-k, no token dropping)
        mixed = torch.zeros_like(embedded)
        for e_idx in range(self.config.num_experts):
            expert_out = self.experts[e_idx](embedded)
            expert_weight = weights[:, e_idx:e_idx + 1]  # (batch, 1)
            mixed = mixed + expert_weight * expert_out

        # 4. Residual connection + projection
        context = self.mixture_proj(mixed) + embedded

        # 5. Decode rooms autoregressively
        rooms = self.decoder(context, num_rooms, teacher_rooms)

        # 6. Store expert weights for visualization
        self._last_expert_weights = self.gating.get_all_weights(embedded).detach()

        return {
            "rooms": rooms,
            "expert_weights": self._last_expert_weights,
            "aux_loss": aux_loss,
        }

    def get_expert_weights(self, constraints: torch.Tensor) -> torch.Tensor:
        """Get expert activation weights for visualization."""
        with torch.no_grad():
            embedded = self.encoder(constraints)
            return self.gating.get_all_weights(embedded)

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
