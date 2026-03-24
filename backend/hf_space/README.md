---
title: Buildify HouseGAN++
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Buildify HouseGAN++ API

HouseGAN++ floor plan layout generator for Buildify.

## API

```
POST /api/predict
{"data": [hg_type_vector, binary_adj, house_w, house_h, num_samples]}
```
