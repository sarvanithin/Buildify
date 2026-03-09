"""
HouseGAN++ integration for Buildify.

Provides spatial floor plan layout generation using a Graph Convolutional
Network trained on the RPLAN dataset of real apartment floor plans.

Public API:
  build_bubble_diagram(constraints) → BubbleDiagram
  generate_layouts(diagram, num_variants, mode) → List[List[Dict]]
"""
from .bubble_diagram import BubbleDiagram, BubbleRoom, build_bubble_diagram, diagram_summary
from .inference import generate_layouts
from .model import HouseGANGenerator, load_pretrained

__all__ = [
    "BubbleDiagram",
    "BubbleRoom",
    "build_bubble_diagram",
    "diagram_summary",
    "generate_layouts",
    "HouseGANGenerator",
    "load_pretrained",
]
