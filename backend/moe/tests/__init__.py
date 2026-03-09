"""
Tests for the Buildify MOE model.
"""
import torch
import pytest
from ..config import MOEConfig
from ..model import BuildifyMOE
from ..data import (
    FloorPlanDataset, encode_constraints, encode_rooms,
    _build_room_list, _solve_layout, STYLE_TEMPLATES
)
from ..experts import create_experts, EXPERT_NAMES
from ..inference import predict_floor_plan, _validate_irc, _resolve_overlaps


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return MOEConfig()


@pytest.fixture
def small_config():
    """Small config for fast tests."""
    c = MOEConfig()
    c.expert_hidden_dims = [64, 32]
    c.embedding_dim = 32
    c.gating_hidden_dim = 32
    c.attention_heads = 2
    c.max_rooms = 10
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModel:
    def test_forward_pass_shape(self, small_config):
        model = BuildifyMOE(small_config)
        x = torch.randn(2, small_config.input_features)
        output = model(x, num_rooms=5)

        assert "rooms" in output
        assert "expert_weights" in output
        assert "aux_loss" in output
        assert output["rooms"]["coords"].shape == (2, 5, 4)
        assert output["rooms"]["type_logits"].shape == (2, 5, small_config.num_room_types)

    def test_gating_weights_sum_to_one(self, small_config):
        model = BuildifyMOE(small_config)
        x = torch.randn(4, small_config.input_features)
        weights = model.get_expert_weights(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_parameter_count(self, small_config):
        model = BuildifyMOE(small_config)
        count = model.count_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_experts_count(self, config):
        experts = create_experts(config)
        assert len(experts) == 8
        assert len(EXPERT_NAMES) == 8

    def test_load_balance_loss_nonzero(self, small_config):
        model = BuildifyMOE(small_config)
        model.train()
        x = torch.randn(8, small_config.input_features)
        output = model(x, num_rooms=3)
        assert output["aux_loss"].item() > 0


# ─────────────────────────────────────────────────────────────────────────────
# Data tests
# ─────────────────────────────────────────────────────────────────────────────

class TestData:
    def test_encode_constraints(self, config):
        vec = encode_constraints(
            bedrooms=3, bathrooms=2, sqft=2000, stories=1, style="modern",
            open_plan=True, primary_suite=True, home_office=False,
            formal_dining=False, garage="2car", laundry="room",
            outdoor="patio", ceiling_height="standard", config=config,
        )
        assert vec.shape == (config.input_features,)
        assert vec.dtype == torch.float32

    def test_build_room_list(self):
        rooms = _build_room_list(
            bedrooms=3, bathrooms=2, sqft=2000, style="modern",
            open_plan=False, primary_suite=True, home_office=True,
            formal_dining=False, garage="2car", laundry="room", outdoor="patio",
        )
        assert len(rooms) > 0
        types = [r["type"] for r in rooms]
        assert "master_bedroom" in types
        assert "kitchen" in types
        assert "living_room" in types

    def test_solve_layout_no_overlap(self):
        rooms = _build_room_list(
            bedrooms=3, bathrooms=2, sqft=2000, style="modern",
            open_plan=False, primary_suite=True, home_office=False,
            formal_dining=False, garage="2car", laundry="room", outdoor="patio",
        )
        placed, w, h = _solve_layout(rooms, 2000, STYLE_TEMPLATES["modern"])
        assert w > 0 and h > 0
        for r in placed:
            assert r["x"] >= 0 and r["y"] >= 0
            assert r["width"] > 0 and r["height"] > 0

    def test_dataset_generation(self, config):
        ds = FloorPlanDataset(num_samples=10, config=config, seed=42)
        assert len(ds) == 10
        sample = ds[0]
        assert "constraints" in sample
        assert "coord_targets" in sample
        assert sample["constraints"].shape == (config.input_features,)


# ─────────────────────────────────────────────────────────────────────────────
# Inference tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInference:
    def test_predict_floor_plan(self):
        constraints = {
            "bedrooms": 3, "bathrooms": 2, "sqft": 2000, "stories": 1,
            "style": "modern", "openPlan": True, "primarySuite": True,
            "homeOffice": False, "formalDining": False, "garage": "2car",
            "laundry": "room", "outdoor": "patio", "ceilingHeight": "standard",
        }
        result = predict_floor_plan(constraints, num_variants=2)
        assert "plans" in result
        assert len(result["plans"]) == 2
        assert "expert_weights" in result
        assert "confidence" in result
        assert result["irc_compliant"]

        for plan in result["plans"]:
            assert len(plan["rooms"]) > 0
            assert plan["totalWidth"] > 0
            assert plan["totalHeight"] > 0

    def test_irc_validation(self):
        rooms = [
            {"type": "bedroom", "x": 0, "y": 0, "width": 5, "height": 5},  # too small
            {"type": "kitchen", "x": 10, "y": 0, "width": 12, "height": 14},
        ]
        validated = _validate_irc(rooms, 50, 40)
        # Bedroom should be scaled up to meet 70 sqft minimum
        assert validated[0]["width"] * validated[0]["height"] >= 70

    def test_overlap_resolution(self):
        rooms = [
            {"type": "living_room", "x": 0, "y": 0, "width": 15, "height": 18},
            {"type": "kitchen", "x": 5, "y": 0, "width": 12, "height": 14},  # overlaps
        ]
        resolved = _resolve_overlaps(rooms, 50, 40)
        # After resolution, rooms should not overlap
        r1, r2 = resolved[0], resolved[1]
        overlaps = (r1["x"] < r2["x"] + r2["width"] and
                   r1["x"] + r1["width"] > r2["x"] and
                   r1["y"] < r2["y"] + r2["height"] and
                   r1["y"] + r1["height"] > r2["y"])
        assert not overlaps, "Rooms should not overlap after resolution"

    def test_grid_snapping(self):
        rooms = [{"type": "bedroom", "x": 3.7, "y": 5.3, "width": 11.5, "height": 12.3}]
        resolved = _resolve_overlaps(rooms, 50, 40)
        # All values should be multiples of 2 (structural grid)
        for r in resolved:
            assert r["x"] % 2 == 0
            assert r["y"] % 2 == 0
            assert r["width"] % 2 == 0
            assert r["height"] % 2 == 0
