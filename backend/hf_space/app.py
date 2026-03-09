"""
Buildify HouseGAN++ — HuggingFace Spaces deployment.

Deploy this to HuggingFace Spaces (ZeroGPU) for free A10G GPU inference.
The Buildify backend calls this endpoint via HOUSEGAN_HF_URL env var.

Deploy steps:
  1. Create new Space at huggingface.co/new-space
     - SDK: Gradio
     - Hardware: ZeroGPU (free, A10G)
     - Name: buildify-housegan
  2. Push this file + requirements.txt to the Space repo
  3. Set HOUSEGAN_HF_URL=https://buildify-housegan.hf.space/api/predict in backend .env

API contract:
  Input:  [hg_type_vector: List[int], binary_adj: List[List[float]],
           house_w: float, house_h: float, num_samples: int]
  Output: [layouts: List[List[List[float]]]]
           layouts[i][j] = [x1, y1, x2, y2] normalised 0-1 for room j in variant i
"""
from __future__ import annotations

import os
import json
import math
import numpy as np

import gradio as gr
import torch
import spaces  # ZeroGPU decorator

# ── Model ─────────────────────────────────────────────────────────────────────

# Inline model definition so the Space is self-contained
# (mirrors moe/housegan/model.py)

NUM_ROOM_TYPES = 18
NOISE_DIM      = 128
GRAPH_DIM      = 128
MASK_SIZE      = 64


class GraphConvLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_fc  = torch.nn.Linear(in_dim, out_dim)
        self.neigh_fc = torch.nn.Linear(in_dim, out_dim)
        self.norm     = torch.nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        agg = torch.matmul(adj / deg, x)
        h = self.self_fc(x) + self.neigh_fc(agg)
        return torch.nn.functional.relu(self.norm(h))


class GraphRelationNetwork(torch.nn.Module):
    def __init__(self, in_dim, hidden=GRAPH_DIM, out_dim=GRAPH_DIM):
        super().__init__()
        self.gc1 = GraphConvLayer(in_dim, hidden)
        self.gc2 = GraphConvLayer(hidden, hidden)
        self.gc3 = GraphConvLayer(hidden, out_dim)

    def forward(self, x, adj):
        return self.gc3(self.gc2(self.gc1(x, adj), adj), adj)


class MaskDecoder(torch.nn.Module):
    def __init__(self, in_dim=GRAPH_DIM):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, 256 * 4 * 4)
        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
            torch.nn.BatchNorm2d(128), torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            torch.nn.BatchNorm2d(64),  torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),
            torch.nn.BatchNorm2d(32),  torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(32, 1, 4, 2, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, z):
        h = torch.nn.functional.relu(self.fc(z)).view(-1, 256, 4, 4)
        return self.deconv(h)


class HouseGANGenerator(torch.nn.Module):
    def __init__(self, num_types=NUM_ROOM_TYPES, noise_dim=NOISE_DIM,
                 graph_dim=GRAPH_DIM, refinement_steps=3):
        super().__init__()
        self.noise_dim = noise_dim
        self.refinement_steps = refinement_steps
        self.type_embed = torch.nn.Embedding(num_types + 1, 64, padding_idx=0)
        self.grn_init   = GraphRelationNetwork(noise_dim + 64, graph_dim, graph_dim)
        self.dec_init   = MaskDecoder(graph_dim)
        self.grn_refine = GraphRelationNetwork(noise_dim + 64 + 4, graph_dim, graph_dim)
        self.dec_refine = MaskDecoder(graph_dim)

    def _mask_stats(self, masks):
        N = masks.size(0)
        flat = masks.view(N, -1)
        grid_y = torch.linspace(0, 1, MASK_SIZE, device=masks.device)
        grid_x = torch.linspace(0, 1, MASK_SIZE, device=masks.device)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing='ij')
        gy, gx = gy.reshape(-1), gx.reshape(-1)
        eps = 1e-6
        total = flat.sum(dim=-1, keepdim=True).clamp(min=eps)
        cx = (flat * gx).sum(-1, keepdim=True) / total
        cy = (flat * gy).sum(-1, keepdim=True) / total
        sx = ((flat * (gx - cx) ** 2).sum(-1, keepdim=True) / total).sqrt()
        sy = ((flat * (gy - cy) ** 2).sum(-1, keepdim=True) / total).sqrt()
        return torch.cat([cx, cy, sx, sy], dim=-1)

    def forward(self, room_types, adj, z=None):
        N = room_types.size(0)
        device = room_types.device
        if z is None:
            z = torch.randn(N, self.noise_dim, device=device)
        type_emb = self.type_embed(room_types)
        x0 = torch.cat([z, type_emb], dim=-1)
        h0 = self.grn_init(x0, adj)
        masks = self.dec_init(h0)
        for _ in range(self.refinement_steps):
            stats = self._mask_stats(masks)
            xr = torch.cat([z, type_emb, stats], dim=-1)
            hr = self.grn_refine(xr, adj)
            masks = self.dec_refine(hr)
        return masks


# ── Load weights ──────────────────────────────────────────────────────────────

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = HouseGANGenerator()
        weights = "housegan_pp.pt"
        if os.path.exists(weights):
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            state = ckpt.get("generator", ckpt.get("model_state_dict", ckpt))
            _model.load_state_dict(state, strict=False)
            print(f"[HouseGAN] Loaded weights from {weights}")
        else:
            print("[HouseGAN] No weights file — using random initialisation (demo mode)")
        _model.eval()
    return _model


# ── Mask → bbox conversion ─────────────────────────────────────────────────────

def masks_to_bboxes(masks_np: np.ndarray, threshold: float = 0.5):
    """Convert (N, 1, 64, 64) masks to normalised (x1,y1,x2,y2) list."""
    bboxes = []
    for mask in masks_np:
        m = (mask[0] >= threshold).astype(np.uint8)
        ys, xs = np.where(m)
        if len(xs) == 0:
            bboxes.append([0.1, 0.1, 0.4, 0.4])   # fallback bbox
        else:
            x1 = float(xs.min()) / MASK_SIZE
            y1 = float(ys.min()) / MASK_SIZE
            x2 = float(xs.max() + 1) / MASK_SIZE
            y2 = float(ys.max() + 1) / MASK_SIZE
            bboxes.append([x1, y1, x2, y2])
    return bboxes


# ── Inference endpoint ────────────────────────────────────────────────────────

@spaces.GPU
def generate_layouts(
    hg_type_vector: list,
    binary_adj: list,
    house_w: float,
    house_h: float,
    num_samples: int = 3,
) -> list:
    """
    Generate floor plan layouts from a bubble diagram.

    Args:
        hg_type_vector: List[int] — HouseGAN room type IDs
        binary_adj:     List[List[float]] — (N,N) adjacency matrix
        house_w:        float — target footprint width in feet
        house_h:        float — target footprint height in feet
        num_samples:    int — number of layout variants to return

    Returns:
        List of layouts, each layout is List of [x1,y1,x2,y2] normalised 0-1
    """
    num_samples = max(1, min(num_samples, 5))
    model = _get_model()

    room_types = torch.tensor(hg_type_vector, dtype=torch.long)
    adj        = torch.tensor(binary_adj, dtype=torch.float32)

    layouts = []
    with torch.no_grad():
        for _ in range(num_samples):
            masks    = model(room_types, adj)          # (N, 1, 64, 64)
            masks_np = masks.cpu().numpy()
            bboxes   = masks_to_bboxes(masks_np)       # List of [x1,y1,x2,y2]
            layouts.append(bboxes)

    return layouts


# ── Gradio UI (also serves as API) ───────────────────────────────────────────

with gr.Blocks(title="Buildify HouseGAN++") as demo:
    gr.Markdown("""
    # Buildify HouseGAN++ — Floor Plan Layout Generator

    Generates spatial floor plan layouts from room adjacency graphs using a
    Graph Convolutional Network.

    **API Usage** (called by Buildify backend):
    ```
    POST /api/predict
    {"data": [hg_type_vector, binary_adj, house_w, house_h, num_samples]}
    ```
    """)

    with gr.Row():
        with gr.Column():
            type_vec_in  = gr.JSON(label="Room Type Vector (List[int])",
                                   value=[1, 2, 3, 4, 5, 13])
            adj_in       = gr.JSON(label="Adjacency Matrix (NxN List[List[float]])",
                                   value=[[0,1,0,0,0,1],[1,0,0,0,0,0],
                                          [0,0,0,1,1,0],[0,0,1,0,0,0],
                                          [0,0,1,0,0,0],[1,0,0,0,0,0]])
            w_in  = gr.Number(label="House Width (ft)", value=46.0)
            h_in  = gr.Number(label="House Height (ft)", value=38.0)
            n_in  = gr.Slider(1, 5, value=3, step=1, label="Num Variants")
            btn   = gr.Button("Generate Layouts", variant="primary")

        with gr.Column():
            out   = gr.JSON(label="Layouts Output (List of variants × rooms)")

    btn.click(
        fn=generate_layouts,
        inputs=[type_vec_in, adj_in, w_in, h_in, n_in],
        outputs=out,
    )

if __name__ == "__main__":
    demo.launch()
