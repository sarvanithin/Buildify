---
title: Buildify HouseGAN++
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: mit
---

# Buildify HouseGAN++ Space

Generates floor plan layouts from room adjacency graphs (bubble diagrams) using
HouseGAN++ — a Graph Convolutional Network trained on the RPLAN dataset.

Used by the [Buildify](https://github.com/sarvanithin/Buildify) backend as a
free ZeroGPU inference endpoint.

## API

```python
import httpx

response = httpx.post(
    "https://buildify-housegan.hf.space/api/predict",
    json={"data": [
        [1, 2, 3, 4, 5, 13],          # room type IDs (HouseGAN format)
        [[0,1,0],[1,0,1],[0,1,0]],     # adjacency matrix
        46.0,                           # house width in feet
        38.0,                           # house height in feet
        3,                              # number of layout variants
    ]}
)
layouts = response.json()["data"][0]
# layouts[variant][room] = [x1, y1, x2, y2]  (normalised 0-1)
```
