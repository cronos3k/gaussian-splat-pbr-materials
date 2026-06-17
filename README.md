# Gaussian Splat PBR Materials

Restoring renderer-native PBR material layers from photoreal capture primitives such as Gaussian splats, scanned meshes, and extracted proxy meshes.

The project demonstrates a screen-space to UV-space pipeline:

```text
overlapping RGB + UV captures
-> intrinsic/de-lit albedo
-> RGB region segmentation
-> vision material tags
-> UV-space voting
-> UV occupancy resistance fill
-> PBR atlas generation
```

The core idea is to keep every screen-space prediction tied to a UV/object-coordinate pass. A model can operate on ordinary rendered images, while the result is still written back into renderer-native texture space.

## Contents

- `src/pbr_surface.py` - UV capture baker, material voting, UV resistance fill, and PBR atlas generation.
- `src/pbr_pixelart.py` - material priors and procedural PBR map generation baseline.
- `scripts/render_uv_captures.py` - Blender helper for rendering aligned RGB and UV-coordinate captures.
- `scripts/intrinsic_ab_test.py` - local A/B harness for intrinsic-image backends.
- `paper/` - paper, figures, and PDF explaining the method and validation results.
- `examples/can_opener_resistance_fill/` - derived output from the 32-view can-opener validation pass.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Optional intrinsic-image and segmentation backends require their own model/checkpoint setup:

```powershell
pip install -e ".[intrinsics,segmentation]"
```

## Basic Usage

Bake aligned color/UV captures into one atlas:

```powershell
pbr-surface bake-captures .\captures\my_asset `
  --output .\runs\my_asset\albedo.png `
  --size 1024 `
  --dilate-mode resistance `
  --dilate-resistance 14 `
  --fill-target-mask .\runs\my_asset\uv_occupancy_mask.png `
  --expand-mask
```

Generate PBR maps from the completed atlas:

```powershell
pbr-surface atlas .\runs\my_asset\albedo.png `
  --mask .\runs\my_asset\albedo_mask.png `
  --material-map .\runs\my_asset\materials.png `
  --material-labels .\runs\my_asset\materials.json `
  --output .\runs\my_asset\pbr `
  --force
```

Render 32 overlapping RGB/UV capture pairs in Blender:

```powershell
blender --background --python .\scripts\render_uv_captures.py -- `
  --obj .\assets\my_asset\model.obj `
  --texture .\assets\my_asset\texture.png `
  --out-dir .\captures\my_asset_sphere32 `
  --name my_asset `
  --size 512 `
  --views sphere32 `
  --ortho-scale-mult 1.65
```

## Validation Snapshot

The included can-opener example used 32 overlapping views and a UV occupancy constrained resistance fill:

| Stage | Atlas coverage | Tagged atlas coverage |
|---|---:|---:|
| Direct `sphere32` samples | 2.71% | 2.03% |
| UV occupancy target | 39.34% | n/a |
| 96px resistance fill | 35.89% | 34.84% |

See `paper/gaussian_splat_pbr_paper.html` or `paper/gaussian_splat_pbr_paper.pdf` for the full write-up.

## Author

Gregor Koch

## License

MIT
