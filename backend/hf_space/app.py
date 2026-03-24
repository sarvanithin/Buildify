"""
Buildify HouseGAN++ — HuggingFace Spaces API (FastAPI, no Gradio).

API:
  POST /api/predict
  {"data": [hg_type_vector, binary_adj, house_w, house_h, num_samples]}
  Returns: {"data": [layouts]}
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_ROOM_TYPES = 18
NOISE_DIM      = 128
GRAPH_DIM      = 128
MASK_SIZE      = 64


# ── Model ─────────────────────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_fc  = nn.Linear(in_dim, out_dim)
        self.neigh_fc = nn.Linear(in_dim, out_dim)
        self.norm     = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        agg = torch.matmul(adj / deg, x)
        return F.relu(self.norm(self.self_fc(x) + self.neigh_fc(agg)))


class GraphRelationNetwork(nn.Module):
    def __init__(self, in_dim, hidden=GRAPH_DIM, out_dim=GRAPH_DIM):
        super().__init__()
        self.gc1 = GraphConvLayer(in_dim, hidden)
        self.gc2 = GraphConvLayer(hidden, hidden)
        self.gc3 = GraphConvLayer(hidden, out_dim)

    def forward(self, x, adj):
        return self.gc3(self.gc2(self.gc1(x, adj), adj), adj)


class MaskDecoder(nn.Module):
    def __init__(self, in_dim=GRAPH_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.ConvTranspose2d(32,  1,   4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(F.relu(self.fc(z)).view(-1, 256, 4, 4))


class HouseGANGenerator(nn.Module):
    def __init__(self, num_types=NUM_ROOM_TYPES, noise_dim=NOISE_DIM,
                 graph_dim=GRAPH_DIM, refinement_steps=3):
        super().__init__()
        self.noise_dim = noise_dim
        self.refinement_steps = refinement_steps
        self.type_embed = nn.Embedding(num_types + 1, 64, padding_idx=0)
        self.grn_init   = GraphRelationNetwork(noise_dim + 64, graph_dim, graph_dim)
        self.dec_init   = MaskDecoder(graph_dim)
        self.grn_refine = GraphRelationNetwork(noise_dim + 64 + 4, graph_dim, graph_dim)
        self.dec_refine = MaskDecoder(graph_dim)

    def _mask_stats(self, masks):
        N = masks.size(0)
        flat = masks.view(N, -1)
        gy, gx = torch.meshgrid(
            torch.linspace(0, 1, MASK_SIZE, device=masks.device),
            torch.linspace(0, 1, MASK_SIZE, device=masks.device),
            indexing='ij'
        )
        gy, gx = gy.reshape(-1), gx.reshape(-1)
        total = flat.sum(-1, keepdim=True).clamp(min=1e-6)
        cx = (flat * gx).sum(-1, keepdim=True) / total
        cy = (flat * gy).sum(-1, keepdim=True) / total
        sx = ((flat * (gx - cx) ** 2).sum(-1, keepdim=True) / total).sqrt()
        sy = ((flat * (gy - cy) ** 2).sum(-1, keepdim=True) / total).sqrt()
        return torch.cat([cx, cy, sx, sy], dim=-1)

    def forward(self, room_types, adj, z=None):
        N = room_types.size(0)
        if z is None:
            z = torch.randn(N, self.noise_dim)
        te = self.type_embed(room_types)
        masks = self.dec_init(self.grn_init(torch.cat([z, te], -1), adj))
        for _ in range(self.refinement_steps):
            masks = self.dec_refine(self.grn_refine(
                torch.cat([z, te, self._mask_stats(masks)], -1), adj))
        return masks


# ── Load model ────────────────────────────────────────────────────────────────

_model = None

def get_model():
    global _model
    if _model is None:
        _model = HouseGANGenerator()
        weights = "housegan_pp.pt"
        if os.path.exists(weights):
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            state = ckpt.get("generator", ckpt.get("model_state_dict", ckpt))
            _model.load_state_dict(state, strict=False)
        _model.eval()
    return _model


def masks_to_bboxes(masks_np, threshold=0.5):
    bboxes = []
    for mask in masks_np:
        m = (mask[0] >= threshold).astype(np.uint8)
        ys, xs = np.where(m)
        if len(xs) == 0:
            bboxes.append([0.1, 0.1, 0.4, 0.4])
        else:
            bboxes.append([
                float(xs.min()) / MASK_SIZE, float(ys.min()) / MASK_SIZE,
                float(xs.max() + 1) / MASK_SIZE, float(ys.max() + 1) / MASK_SIZE,
            ])
    return bboxes


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Buildify HouseGAN++")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    data: list


@app.get("/")
def root():
    return {"status": "ok", "service": "Buildify HouseGAN++"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(req: PredictRequest):
    hg_type_vector, binary_adj, house_w, house_h, num_samples = req.data
    num_samples = max(1, min(int(num_samples), 5))

    model = get_model()
    room_types = torch.tensor(hg_type_vector, dtype=torch.long)
    adj        = torch.tensor(binary_adj, dtype=torch.float32)

    layouts = []
    with torch.no_grad():
        for _ in range(num_samples):
            masks  = model(room_types, adj)
            bboxes = masks_to_bboxes(masks.numpy())
            layouts.append(bboxes)

    return {"data": [layouts]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
