# Academia Upload Metadata

Use this file to fill Academia's upload form for the publication PDF.

## File

`paper/gaussian_splat_pbr_publication.pdf`

## Title

UV-Guided Recovery of PBR Material Atlases from Photoreal Capture Primitives

## Author

Gregor Hubert Max Koch

## Abstract

Photoreal capture methods such as 3D Gaussian splatting and photogrammetry can reproduce scenes with high visual fidelity, yet their appearance representation is usually not a renderer-native material model. Captured colors often mix base color, shadows, highlights, view-dependent response, and exposure. As a result, assets that look plausible under their original capture illumination can fail when relit in physically based renderers. This paper presents a prototype pipeline for converting paired RGB and UV-coordinate screen captures into physically based rendering (PBR) texture atlases. The method renders overlapping RGB/UV views, applies intrinsic-image decomposition to estimate de-lit appearance, segments and semantically tags material regions from the original RGB observations, votes those observations into UV texture space, and completes sparse atlas regions using a UV-occupancy-constrained resistance fill. A validation run on a multi-material scanned can opener shows that direct 32-view UV voting is more reliable than a single view but remains sparse: direct atlas coverage is 2.71%, with 2.03% tagged material coverage. Applying a 96-pixel resistance fill constrained by an inferred UV occupancy mask expands usable atlas coverage to 35.89% and tagged material coverage to 34.84%, while avoiding empty atlas padding. The result is not a calibrated inverse-rendering solution, but it demonstrates a practical bridge from capture-native appearance to conventional albedo, material-label, normal, height, and occlusion/roughness/metallicity maps.

## Research Interests / Tags

- Computer Graphics
- Physically Based Rendering
- Gaussian Splatting
- Photogrammetry
- Texture Synthesis
- Intrinsic Images
- Material Estimation
- Neural Rendering
- Game Development
- 3D Reconstruction

## Short Post Text

This preprint describes a practical RGB/UV capture pipeline for recovering renderer-native PBR material atlases from Gaussian splats, scanned meshes, and extracted proxy meshes. The central idea is to preserve UV correspondence for every screen-space prediction, then combine intrinsic de-lighting, material tagging, multiview UV voting, and UV-occupancy-constrained resistance fill into conventional albedo, material-label, normal, height, and ORM maps.

## Code URL

https://github.com/cronos3k/gaussian-splat-pbr-materials
