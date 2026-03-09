"""
MOE model hyperparameters and tier configuration.
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MOEConfig:
    # ── Model architecture ──────────────────────────────────────────────────
    num_experts: int = 8
    expert_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    gating_hidden_dim: int = 256
    top_k_experts: int = 4          # sparse routing: activate top-4 of 8
    attention_heads: int = 4
    embedding_dim: int = 128
    input_features: int = 20        # encoded constraint vector size
    max_rooms: int = 20
    room_features: int = 7          # x, y, width, height, type_idx, zone, is_exterior

    # ── Training ────────────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 200
    early_stop_patience: int = 20
    load_balance_weight: float = 0.1  # auxiliary loss weight for expert balancing
    compliance_weight: float = 0.3    # weight for IRC compliance loss
    adjacency_weight: float = 0.2     # weight for adjacency reward loss
    dropout: float = 0.1

    # ── Data ────────────────────────────────────────────────────────────────
    num_train_samples: int = 100_000
    num_val_samples: int = 10_000
    num_test_samples: int = 10_000
    augmentation_jitter: float = 0.10  # ±10% size jitter

    # ── Room types (index → name) ───────────────────────────────────────────
    ROOM_TYPES: list = field(default_factory=lambda: [
        "living_room", "kitchen", "dining_room", "family_room",
        "master_bedroom", "bedroom", "ensuite_bathroom", "bathroom",
        "half_bath", "hallway", "foyer", "home_office",
        "laundry_room", "garage", "walk_in_closet", "closet",
        "pantry", "mudroom", "utility_room",
        "patio", "deck",
    ])

    # ── Styles (index → name) ──────────────────────────────────────────────
    STYLES: list = field(default_factory=lambda: [
        "modern", "traditional", "craftsman", "ranch",
        "farmhouse", "contemporary", "colonial", "cape_cod",
    ])

    # ── Tier limits ─────────────────────────────────────────────────────────
    TIER_LIMITS: Dict[str, int] = field(default_factory=lambda: {
        "free": 5,           # 5 generations per day
        "pro": -1,           # unlimited
        "enterprise": -1,    # unlimited + API access
    })

    TIER_VARIANTS: Dict[str, int] = field(default_factory=lambda: {
        "free": 1,
        "pro": 3,
        "enterprise": 5,
    })

    # ── Weights path ────────────────────────────────────────────────────────
    weights_dir: str = "moe/weights"
    model_filename: str = "buildify_moe.pt"

    @property
    def num_room_types(self) -> int:
        return len(self.ROOM_TYPES)

    @property
    def num_styles(self) -> int:
        return len(self.STYLES)

    @property
    def output_dim(self) -> int:
        """Dimension per room: x, y, w, h, type_logits, zone, exterior."""
        return 4 + self.num_room_types + 2  # 4 coords + type probs + zone + exterior
