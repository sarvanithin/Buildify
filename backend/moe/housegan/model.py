"""
HouseGAN++ model architecture — Graph Convolutional Network for floor plan generation.

Based on: "HouseGAN++: Generative Adversarial Layout Refinement Network
           towards Intelligent Computational Agent for Professional Architects"
           (Nauata et al., 2021)

Architecture:
  - Graph Relation Network (GRN): message-passing between room nodes
  - Convolutional decoder: generates 64×64 binary mask per room
  - Refinement: iterative layout improvement

Input:
  - z:          (N, noise_dim) noise vectors per room
  - room_types: (N, num_types) one-hot type embeddings
  - adj_matrix: (N, N) binary adjacency

Output:
  - masks:      (N, 1, 64, 64) binary room masks
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────
NUM_ROOM_TYPES = 18      # RPLAN + our US extensions
NOISE_DIM      = 128
GRAPH_DIM      = 128
MASK_SIZE      = 64


# ── Graph Relation Network ────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """Single graph convolution: aggregates neighbor features via adjacency."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_fc   = nn.Linear(in_dim, out_dim)
        self.neigh_fc  = nn.Linear(in_dim, out_dim)
        self.norm      = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   (N, in_dim)
        adj: (N, N) binary / weighted adjacency
        """
        # Degree-normalise
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        agg = torch.matmul(adj / deg, x)   # (N, in_dim)

        h = self.self_fc(x) + self.neigh_fc(agg)
        return F.relu(self.norm(h))


class GraphRelationNetwork(nn.Module):
    """
    3-layer GCN that encodes room-level context.
    Each room node aggregates information from its neighbours
    (adjacent rooms) to produce a spatially-aware feature vector.
    """

    def __init__(self, in_dim: int = NOISE_DIM + NUM_ROOM_TYPES,
                 hidden: int = GRAPH_DIM, out_dim: int = GRAPH_DIM):
        super().__init__()
        self.gc1 = GraphConvLayer(in_dim,  hidden)
        self.gc2 = GraphConvLayer(hidden,  hidden)
        self.gc3 = GraphConvLayer(hidden,  out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.gc1(x, adj)
        h = self.gc2(h, adj)
        h = self.gc3(h, adj)
        return h   # (N, out_dim)


# ── Convolutional Mask Decoder ────────────────────────────────────────────────

class MaskDecoder(nn.Module):
    """
    Decodes a per-room feature vector into a 64×64 binary occupancy mask.
    Uses transpose convolution (upsampling) from 4×4 → 64×64.
    """

    def __init__(self, in_dim: int = GRAPH_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            # 4×4 → 8×8
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 8×8 → 16×16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 16×16 → 32×32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32×32 → 64×64
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, in_dim) → masks: (N, 1, 64, 64)"""
        h = F.relu(self.fc(z)).view(-1, 256, 4, 4)
        return self.deconv(h)


# ── Full HouseGAN++ Generator ─────────────────────────────────────────────────

class HouseGANGenerator(nn.Module):
    """
    Full HouseGAN++ generator.

    Combines:
    1. Type embedding
    2. Graph relation network (spatial context)
    3. Mask decoder (64×64 room footprints)
    4. Iterative refinement (re-run GCN with generated mask features)
    """

    def __init__(self,
                 num_types: int = NUM_ROOM_TYPES,
                 noise_dim: int = NOISE_DIM,
                 graph_dim: int = GRAPH_DIM,
                 refinement_steps: int = 3):
        super().__init__()
        self.noise_dim = noise_dim
        self.refinement_steps = refinement_steps

        self.type_embed = nn.Embedding(num_types + 1, 64, padding_idx=0)

        # Initial GRN: noise + type → graph features
        self.grn_init = GraphRelationNetwork(
            in_dim=noise_dim + 64,
            hidden=graph_dim,
            out_dim=graph_dim,
        )
        self.decoder_init = MaskDecoder(graph_dim)

        # Refinement GRN: adds mask statistics to node features
        self.grn_refine = GraphRelationNetwork(
            in_dim=noise_dim + 64 + 4,   # +4 mask stats (mean, std, centroid x/y)
            hidden=graph_dim,
            out_dim=graph_dim,
        )
        self.decoder_refine = MaskDecoder(graph_dim)

    def _mask_stats(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute per-room spatial statistics from masks.
        masks: (N, 1, 64, 64)
        returns: (N, 4) — [mean_x, mean_y, std_x, std_y] in [0,1]
        """
        N = masks.size(0)
        flat = masks.view(N, -1)  # (N, 64*64)

        grid_y = torch.linspace(0, 1, MASK_SIZE, device=masks.device)
        grid_x = torch.linspace(0, 1, MASK_SIZE, device=masks.device)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing='ij')
        gy = gy.reshape(-1)  # (64*64,)
        gx = gx.reshape(-1)

        eps = 1e-6
        total = flat.sum(dim=-1, keepdim=True).clamp(min=eps)
        cx = (flat * gx.unsqueeze(0)).sum(-1, keepdim=True) / total
        cy = (flat * gy.unsqueeze(0)).sum(-1, keepdim=True) / total
        sx = ((flat * (gx.unsqueeze(0) - cx) ** 2).sum(-1, keepdim=True) / total).sqrt()
        sy = ((flat * (gy.unsqueeze(0) - cy) ** 2).sum(-1, keepdim=True) / total).sqrt()

        return torch.cat([cx, cy, sx, sy], dim=-1)  # (N, 4)

    def forward(self, room_types: torch.Tensor, adj: torch.Tensor,
                z: torch.Tensor = None) -> torch.Tensor:
        """
        room_types: (N,) int — room type IDs
        adj:        (N, N) float — adjacency matrix
        z:          (N, noise_dim) float — noise (sampled if None)

        Returns:
          masks: (N, 1, 64, 64) — room occupancy masks
        """
        N = room_types.size(0)
        device = room_types.device

        if z is None:
            z = torch.randn(N, self.noise_dim, device=device)

        type_emb = self.type_embed(room_types)   # (N, 64)

        # Initial pass
        x0 = torch.cat([z, type_emb], dim=-1)   # (N, noise+64)
        h0 = self.grn_init(x0, adj)
        masks = self.decoder_init(h0)

        # Refinement passes
        for _ in range(self.refinement_steps):
            stats = self._mask_stats(masks)                # (N, 4)
            xr = torch.cat([z, type_emb, stats], dim=-1)  # (N, noise+64+4)
            hr = self.grn_refine(xr, adj)
            masks = self.decoder_refine(hr)

        return masks  # (N, 1, 64, 64)


# ── Weight loading ────────────────────────────────────────────────────────────

def load_pretrained(weights_path: str, device: str = "cpu") -> HouseGANGenerator:
    """Load pre-trained HouseGAN++ weights."""
    model = HouseGANGenerator()
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    # Handle both raw state_dict and checkpoint dicts
    state = ckpt.get("generator", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[HouseGAN] Loaded weights from {weights_path}")
    return model
