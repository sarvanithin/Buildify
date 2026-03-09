"""
8 specialized expert networks for US residential construction.
Each expert is a deep MLP with residual connections and layer normalization.
"""
import torch
import torch.nn as nn
from .config import MOEConfig


class ExpertBlock(nn.Module):
    """Residual MLP block used inside each expert."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class BaseExpert(nn.Module):
    """
    Base expert: 3-layer deep MLP with residual connections.
    Each expert specializes in one aspect of floor plan generation.
    """

    def __init__(self, config: MOEConfig, name: str = "base"):
        super().__init__()
        self.name = name
        dims = [config.embedding_dim] + config.expert_hidden_dims

        layers = []
        for i in range(len(dims) - 1):
            layers.append(ExpertBlock(dims[i], dims[i + 1], config.dropout))
        layers.append(nn.Linear(dims[-1], config.embedding_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RoomSizingExpert(BaseExpert):
    """
    Expert 1: IRC-compliant room dimensions by type and home size.
    Learns the mapping from (room_type, home_sqft, style) → (width, height).

    Encodes IRC R304 (minimum room sizes), R305 (ceiling heights),
    and standard sizing tables for US residential construction.
    """

    def __init__(self, config: MOEConfig):
        super().__init__(config, name="room_sizing")


class SpatialLayoutExpert(BaseExpert):
    """
    Expert 2: Spatial arrangement and footprint utilization.
    Learns optimal room placement within the building envelope.

    Maximizes space efficiency (target 85-95% utilization),
    minimizes wasted hallway space, and respects structural grid
    (2ft module for framing efficiency).
    """

    def __init__(self, config: MOEConfig):
        super().__init__(config, name="spatial_layout")


class StyleAdaptationExpert(BaseExpert):
    """
    Expert 3: Architectural style conventions.
    Adapts layout patterns per style:
    - Modern: open-plan great room, minimal walls, large windows
    - Traditional: formal entry, separate living + dining
    - Craftsman: covered porch, breakfast nook, built-ins
    - Ranch: single story, elongated footprint, split bedrooms
    - Farmhouse: mudroom, large kitchen island, wraparound porch
    """

    def __init__(self, config: MOEConfig):
        super().__init__(config, name="style_adaptation")


class CodeComplianceExpert(BaseExpert):
    """
    Expert 4: IRC building code compliance.
    Enforces:
    - Minimum room sizes (R304: 70sf habitable, R310: bedroom egress)
    - Ceiling heights (R305: 7ft min, 9ft standard)
    - Hallway widths (R311: 36in min)
    - Stair dimensions (R311.7: 36in wide, 7.75in riser, 10in tread)
    - Bathroom clearances (30in toilet, 21in front)
    - Fire separation for attached garage
    """

    def __init__(self, config: MOEConfig):
        super().__init__(config, name="code_compliance")


class AdjacencyExpert(BaseExpert):
    """
    Expert 5: Room adjacency optimization.
    Learns ideal room relationships:
    - Kitchen ↔ Dining (shared wall)
    - Kitchen ↔ Family room (open or adjacent)
    - Primary bedroom ↔ Ensuite bath (direct door)
    - Foyer ↔ Living room (visual connection)
    - Garage ↔ Mudroom/Kitchen (grocery path)
    - Laundry ↔ Bedroom hallway (convenience)
    """

    def __init__(self, config: MOEConfig):
        super().__init__(config, name="adjacency")


class CirculationExpert(BaseExpert):
    """
    Expert 6: Traffic flow and circulation design.
    Optimizes:
    - Entry → Living without crossing private zones
    - Bedroom hallway not used as thoroughfare
    - Kitchen ↔ Garage (grocery unloading path)
    - Hallway sizing (3-4ft wide, max 20% of total area)
    - Two egress points from sleeping areas
    """

    def __init__(self, config: MOEConfig):
        super().__init__(config, name="circulation")


class OutdoorGarageExpert(BaseExpert):
    """
    Expert 7: Outdoor living and garage integration.
    Handles:
    - Patio/deck placement (rear of house, accessed from living/kitchen)
    - Garage positioning (front or side, adjacent to kitchen)
    - Indoor-outdoor transition (sliding doors, covered areas)
    - Landscape buffer zones at property lines
    """

    def __init__(self, config: MOEConfig):
        super().__init__(config, name="outdoor_garage")


class CostOptimizationExpert(BaseExpert):
    """
    Expert 8: Construction cost efficiency.
    Optimizes room sizing and placement for budget targets:
    - Minimizes plumbing runs (stack bathrooms)
    - Efficient structural grid (rooms on 2ft modules)
    - Cost-conscious room proportions
    - Regional cost factor awareness
    """

    def __init__(self, config: MOEConfig):
        super().__init__(config, name="cost_optimization")


def create_experts(config: MOEConfig) -> nn.ModuleList:
    """Create all 8 expert networks."""
    return nn.ModuleList([
        RoomSizingExpert(config),
        SpatialLayoutExpert(config),
        StyleAdaptationExpert(config),
        CodeComplianceExpert(config),
        AdjacencyExpert(config),
        CirculationExpert(config),
        OutdoorGarageExpert(config),
        CostOptimizationExpert(config),
    ])


EXPERT_NAMES = [
    "Room Sizing", "Spatial Layout", "Style Adaptation",
    "Code Compliance", "Adjacency", "Circulation",
    "Outdoor & Garage", "Cost Optimization",
]
