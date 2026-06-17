# Gaussian Splat PBR Materials

Restoring renderer-native PBR material layers from photoreal capture primitives such as Gaussian splats, scanned meshes, and extracted proxy meshes.

**Author:** Gregor Hubert Max Koch  
**Publication paper:** [HTML](paper/gaussian_splat_pbr_publication.html) | [PDF](paper/gaussian_splat_pbr_publication.pdf)  
**Development paper:** [HTML](paper/gaussian_splat_pbr_paper.html) | [PDF](paper/gaussian_splat_pbr_paper.pdf)

![UV-space resistance fill result](paper/figures/figure_09_uv_resistance_fill.png)

## Abstract

Gaussian splatting and photogrammetry can reproduce a scene with high photographic fidelity, but the captured appearance is usually baked radiance rather than a renderer-native material model. When such assets are placed in Unreal Engine, Blender, WebGPU, or another physically based renderer, relighting often fails because albedo, roughness, metallicity, normals, height, and semantic material categories are absent or unreliable.

This project converts paired screen captures into UV-space PBR material atlases. The key constraint is to preserve the relationship between screen-space observations and texture coordinates: every intrinsic-image prediction and every material tag is projected back through a UV/object-coordinate pass. The current best path uses split-source processing plus overlapping multiview capture: use intrinsic decomposition for de-lit texture values, tag materials from original RGB views, vote all labels into UV space, then complete the sparse atlas with a UV-occupancy-constrained resistance fill.

## Why This Matters

Photoreal capture primitives are strong at reproducing the lighting conditions under which they were observed. That strength becomes a weakness when an application needs new lights, edited materials, dynamic time of day, gameplay illumination, or physically plausible asset reuse.

Gaussian splats encode radiance-like appearance in point or ellipsoid attributes. Scanned meshes often carry a single baked diffuse texture. Extracted surfaces from splats or dense captures may inherit colors that include shadows, highlights, and camera exposure. A renderer still needs material parameters:

| Needed by renderer | Typical capture gives |
|---|---|
| Base color / albedo | Baked color with shadows and highlights |
| Roughness | Usually missing |
| Metallic / specular class | Usually missing |
| Height and normal detail | Usually missing or geometry-bound |
| Material identity | Usually implicit in RGB only |

The practical question is:

> Can we take a Gaussian splat, extracted mesh, or scanned textured mesh, screenshot it in world space while preserving texture/UV correspondence, and recover a usable PBR material layer stack?

This repository contains a working prototype for that path.

## Method

The pipeline keeps every screen-space prediction grounded by an explicit UV pass.

```text
32x overlapping RGB + UV captures
-> intrinsic/de-lit albedo
-> RGB region segmentation
-> vision material tags
-> UV-space material voting
-> UV occupancy resistance fill
-> PBR atlas generation
```

![Pipeline overview](paper/figures/figure_01_pipeline.png)

| Stage | Input | Output | Purpose |
|---|---|---|---|
| Render/capture | Mesh, splat-derived mesh, or scanned asset | `*_color.png`, `*_uv.png` | Capture visible appearance and texture-coordinate relation. |
| Intrinsic/de-lighting | Color capture | Albedo candidate, optional shading/roughness/metal/specular | Reduce lighting leakage in material decisions. |
| Region segmentation | Source RGB capture or external masks | Material-relevant regions | Split visible surface regions before tagging. |
| Material tagging | RGB region crops | Semantic material labels | Convert visual evidence into PBR priors. |
| UV vote | De-lit albedo, RGB labels, UV pass | UV-space albedo atlas and material map | Preserve renderer-native texture coordinates. |
| UV completion | Sparse albedo, material map, UV occupancy mask | Completed atlas domain | Fill voids while resisting texture edges and empty atlas padding. |
| PBR synthesis | Completed albedo + material labels | Normal, height, ORM | Generate renderer-consumable material maps. |

## Key Finding: Split Sources Matter

The first multi-material tests failed when material tagging was done from a de-lit albedo image. The de-lighting pass removed some semantic color cues that the material tagger needed. The corrected pipeline tags materials from the original RGB capture, but writes de-lit albedo values into the atlas.

![Split-source material tagging](paper/figures/figure_07_split_source_tagging.png)

## Key Finding: One View Is Not Enough

Single-view examples are useful diagnostics, but they are not a complete solution. They are front-view biased, miss occluded materials, and write very sparse UV atlases. The current validation path renders 32 overlapping RGB/UV capture pairs from a Fibonacci sphere around the asset.

![32-view material voting](paper/figures/figure_08_multiview_voting.png)

On the can opener, the single-view material atlas was heavily front-view biased: 9,158 tagged atlas pixels voted `plastic` and only 416 voted `metal_steel`. The 32-view pass used 384 region-level material calls with zero fallbacks and changed the UV vote to 14,973 `metal_steel`, 6,008 `plastic`, and 287 `unknown` pixels.

| Capture protocol | Capture pairs | Region calls | Fallbacks | Direct atlas coverage | Tagged atlas coverage | Main issue |
|---|---:|---:|---:|---:|---:|---|
| Single front view | 1 | 20 | 0 | 0.91% | 0.91% | Front-view material bias |
| `sphere32` multiview | 32 | 384 | 0 | 2.71% | 2.03% | Better vote, still sparse |

## UV-Space Resistance Fill

The multiview vote gives higher-confidence samples, but it is still sparse because many atlas texels are never hit by a screen pixel. A naive dilation can fill those holes, but it also spreads across empty atlas padding and material boundaries.

The implemented fix is a resistance-weighted UV flood constrained by a target mask. The target mask represents known UV island territory. For the scanned can opener we derived it from non-empty source texture texels. For splat-derived meshes, the same mask should come from a UV occupancy rasterization of the extracted triangles.

For each missing texel, neighboring samples vote with a weight proportional to:

```text
exp(-resistance * color_difference)
```

This allows fills to continue through smooth texture regions while resisting strong albedo or material-edge changes. Material IDs are expanded with the same guide, so labels spread through compatible UV neighborhoods instead of simple nearest-neighbor padding.

| UV completion stage | Atlas coverage | Tagged atlas coverage | Notes |
|---|---:|---:|---|
| Direct `sphere32` samples | 2.71% | 2.03% | Trusted but too sparse for renderer export |
| UV occupancy target | 39.34% | n/a | Estimated island domain from the scan texture |
| 96px resistance fill | 35.89% | 34.84% | Covers 91.23% of the target without filling empty atlas padding |

The included derived example is in [`examples/can_opener_resistance_fill`](examples/can_opener_resistance_fill).

## PBR Outputs

The completed material map is combined with a de-lit albedo atlas and converted into renderer-style maps:

| Map | Role |
|---|---|
| `source_baked_albedo.png` | De-lit UV-space base color candidate |
| `source_baked_albedo_materials.png` | Integer material-label atlas |
| `pbr/source_baked_albedo_n.png` | Synthesized tangent-space normal map |
| `pbr/source_baked_albedo_h.png` | Height map for parallax/displacement-style workflows |
| `pbr/source_baked_albedo_orm.png` | Packed occlusion, roughness, metallicity map |

## Repository Layout

- [`src/pbr_surface.py`](src/pbr_surface.py) - UV capture baker, material voting, UV resistance fill, and PBR atlas generation.
- [`src/pbr_pixelart.py`](src/pbr_pixelart.py) - material priors and procedural PBR map generation baseline.
- [`scripts/render_uv_captures.py`](scripts/render_uv_captures.py) - Blender helper for rendering aligned RGB and UV-coordinate captures.
- [`scripts/intrinsic_ab_test.py`](scripts/intrinsic_ab_test.py) - local A/B harness for intrinsic-image backends.
- [`paper/`](paper) - full paper, figures, and PDF.
- [`examples/can_opener_resistance_fill/`](examples/can_opener_resistance_fill) - derived output from the 32-view validation pass.

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

## Limitations

This is a prototype validation, not a full benchmark. The generated roughness, metallicity, height, and normal maps are plausible renderer inputs rather than measured ground truth. Material labels can fail on ambiguous or mixed-material crops. Gaussian splat assets still need a robust way to render or derive coordinate/UV passes and UV occupancy masks.

## References

See the [paper](paper/gaussian_splat_pbr_paper.md) for references covering 3D Gaussian Splatting, physically based shading, intrinsic image decomposition, Segment Anything, Google Scanned Objects, and the procedural PBR baseline.

## License

GNU Affero General Public License v3.0 or later (`AGPL-3.0-or-later`).

The intent of this license choice is strong copyleft. If a third party modifies,
redistributes, or runs this project as a network-accessible service, the
corresponding source code for the covered work and its modifications must remain
available under the same license. See [`LICENSE`](LICENSE) and
[`LICENSE-NOTICE.md`](LICENSE-NOTICE.md).
