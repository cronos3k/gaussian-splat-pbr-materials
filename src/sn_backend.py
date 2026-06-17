# Copyright (C) 2026 Gregor Koch
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
sn_backend.py -- optional StableNormal diffusion normal-map backend.

Lazy: torch/diffusers are only imported when load_predictor() is called, so the core
procedural pipeline stays dependency-light. Seamless-aware: a tile is wrap-tiled 3x3
before upscaling so the diffusion model "sees" the wrapped neighbours, then only the
centre tile is kept -> the result still tiles. All caches are forced to a LOCAL dir.

NOTE: for tiny pixel art (e.g. 32x32) the default procedural normals are crisper and
preserve seams/detail better; StableNormal tends to flatten them. Use this only for
larger (512px+) textures. Install deps with: pip install -r requirements-diffusion.txt

StableNormal API (from repo hubconf.py):
  predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", ...)
  normal_pil = predictor(img, resolution=R, match_input_resolution=True, data_type="indoor")
"""
import os
import numpy as np
from PIL import Image

_PREDICTOR = None


def load_predictor(cache_dir, device="cuda:0", turbo=True):
    """Load (once) and return the StableNormal predictor. Downloads weights on first call."""
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    # Keep every cache local to the project folder.
    os.environ["HF_HOME"] = os.path.join(cache_dir, "hf")
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hf", "hub")
    import sys
    import importlib
    import torch
    from huggingface_hub import snapshot_download
    torch.hub.set_dir(os.path.join(cache_dir, "torchhub"))

    # Compat shim: StableNormal's custom pipeline imports the pre-0.27 module path
    # `diffusers.models.controlnet`, which newer diffusers (0.37) moved to
    # `diffusers.models.controlnets.controlnet`. Alias it so the dynamic module loads.
    try:
        importlib.import_module("diffusers.models.controlnet")
    except ModuleNotFoundError:
        new = importlib.import_module("diffusers.models.controlnets.controlnet")
        sys.modules["diffusers.models.controlnet"] = new

    # hubconf does os.path.join(local_cache_dir, version) and feeds it to
    # diffusers.from_pretrained. On Windows the default "Stable-X" prefix becomes an
    # invalid repo id ("Stable-X\\..."). So: pre-fetch each weight repo into a local
    # folder, then point local_cache_dir there -> from_pretrained loads from disk.
    weights = os.path.join(cache_dir, "weights")
    versions = ["yoso-normal-v0-3"] if turbo else ["yoso-normal-v0-3", "stable-normal-v0-1"]
    for v in versions:
        dest = os.path.join(weights, v)
        if not (os.path.isdir(dest) and os.listdir(dest)):
            print(f"  downloading Stable-X/{v} -> {dest}")
            snapshot_download(repo_id=f"Stable-X/{v}", local_dir=dest)

    entry = "StableNormal_turbo" if turbo else "StableNormal"
    _PREDICTOR = torch.hub.load(
        "Stable-X/StableNormal", entry, trust_repo=True, device=device,
        local_cache_dir=weights,
    )
    return _PREDICTOR


def diffusion_normal(img_rgb, target_res=512, directx=False, seamless=True):
    """NxN diffuse PIL -> NxN normal map (uint8 HxWx3). Requires load_predictor() first."""
    if _PREDICTOR is None:
        raise RuntimeError("call load_predictor() before diffusion_normal()")
    W, H = img_rgb.size
    rgb = img_rgb.convert("RGB")

    if seamless:
        tiled = Image.fromarray(np.tile(np.array(rgb), (3, 3, 1)))      # wrap 3x3
        up = tiled.resize((target_res, target_res), Image.BICUBIC)
        nmap = _PREDICTOR(up, resolution=target_res, match_input_resolution=True,
                          data_type="indoor").convert("RGB")
        full = nmap.resize((W * 3, H * 3), Image.LANCZOS)
        center = full.crop((W, H, 2 * W, 2 * H))                        # keep centre tile
    else:
        up = rgb.resize((target_res, target_res), Image.BICUBIC)
        nmap = _PREDICTOR(up, resolution=target_res, match_input_resolution=True,
                          data_type="indoor").convert("RGB")
        center = nmap.resize((W, H), Image.LANCZOS)

    v = np.array(center).astype(np.float32) / 255.0 * 2.0 - 1.0         # decode
    n = np.linalg.norm(v, axis=2, keepdims=True)
    n[n == 0] = 1.0
    v = v / n                                                           # renormalize
    if directx:
        v[..., 1] = -v[..., 1]                                          # OpenGL->DirectX green
    return ((v * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
