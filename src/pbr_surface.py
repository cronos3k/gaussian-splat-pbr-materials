#!/usr/bin/env python3
"""
pbr_surface -- generate plausible PBR maps from baked diffuse/base-color UV atlases.

This is the mesh/Gaussian-splat companion to pbr_pixelart.py. It keeps the same core
split:

  * deterministic color clusters are the spatial masks,
  * a vision model only assigns material semantics to those clusters,
  * procedural image math writes height, tangent-space normal, and packed ORM maps.

Unlike the pixel-art path, this file defaults to non-wrapping, mask-aware filtering so
UV islands do not bleed across atlas borders. It also includes a small UV-pass capture
baker: render color plus a UV-coordinate material from any engine, then project the
screen pixels back into a texture atlas before generating PBR maps.
"""

import argparse
import base64
from collections import deque
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw
from tqdm import tqdm

from pbr_pixelart import (
    MATERIALS,
    MATERIAL_NAMES,
    RELIEF_SCALE,
    RAISED_FEATURES,
    SUNKEN_FEATURES,
    SWATCH,
    apply_filename_prior,
    clean_name_hint,
    copy_res,
    extract_palette,
    extract_json_object,
    file_hash,
    heuristic_result,
    load_manifest,
    reset_context,
    resolve_model,
)


IMAGE_EXTS = {".png", ".bmp", ".gif", ".tga", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
MATERIAL_TO_INDEX = {name: i + 1 for i, name in enumerate(MATERIAL_NAMES)}
INDEX_TO_MATERIAL = {i + 1: name for i, name in enumerate(MATERIAL_NAMES)}

SURFACE_SYSTEM_PROMPT = (
    "You analyze one diffuse/base-color texture or UV atlas for a PBR material pipeline. "
    "It may be baked from photogrammetry, a Gaussian-splat proxy mesh, marching-cubes "
    "geometry, or a normal mesh. The image can contain baked lighting and shadows. The "
    "numbered swatches below the texture are deterministic color clusters with coverage. "
    "Your job is to assign each swatch a material; do not invent per-pixel masks.\n"
    "IMPORTANT: shadowed and lit versions of the same surface should usually keep the "
    "same material. Different colors can still be different materials: exposed metal on "
    "paint, moss on stone, grout between tiles, soil under grass, etc.\n"
    "Manufactured object prior: handles, housings, buttons, packaging, painted parts, "
    "and red/black/blue grip surfaces are usually plastic, leather, cloth, paint over "
    "metal, or rubber-like unknown; do not use skin unless an actual human/animal body "
    "surface is visible.\n"
    "Return ONE JSON object, no prose, with these keys:\n"
    '  "palette": object mapping each swatch number (string) to a material from the '
    "ALLOWED list.\n"
    '  "material": dominant material from ALLOWED.\n'
    '  "pattern": one of [planks, bricks, tiles, cobble, scales, grains, crystals, '
    "noise, smooth, woven, organic].\n"
    '  "grain": dominant line/seam direction in texture space, one of '
    "[horizontal, vertical, diagonal, none].\n"
    '  "relief": overall bumpiness, one of [flat, low, medium, high].\n'
    '  "edges": whether boundaries between material/color clusters are recessed grooves, '
    "raised ridges, or neither -- one of [recessed, raised, none].\n"
    '  "features": array (<=6) of obvious localized points, each '
    '{"type":..., "x":int, "y":int}; type in [rivet, stud, bump, gem, knot, crack, '
    "hole, scratch, highlight]. Use [] if unsure.\n"
    "ALLOWED materials: " + ", ".join(MATERIAL_NAMES) + "\n"
    'Example: {"palette":{"0":"stone","1":"brick"},"material":"brick",'
    '"pattern":"bricks","grain":"horizontal","relief":"medium","edges":"recessed",'
    '"features":[]}'
)

REGION_SYSTEM_PROMPT = (
    "You analyze one masked region from a rendered surface for a PBR material pipeline. "
    "The image may come from photogrammetry, a Gaussian-splat proxy mesh, marching-cubes "
    "geometry, or a normal mesh. Only classify the visible unmasked surface region; ignore "
    "checkerboard/background pixels and ignore unrelated hidden areas. The numbered swatches "
    "below the image are deterministic colors sampled only from this masked region.\n"
    "Manufactured object prior: handles, housings, buttons, packaging, painted parts, "
    "and red/black/blue grip surfaces are usually plastic, leather, cloth, paint over "
    "metal, or rubber-like unknown; do not use skin unless an actual human/animal body "
    "surface is visible.\n"
    "Return ONE JSON object, no prose, with these keys:\n"
    '  "palette": object mapping each swatch number (string) to a material from the '
    "ALLOWED list.\n"
    '  "material": dominant material from ALLOWED for the masked region.\n'
    '  "pattern": one of [planks, bricks, tiles, cobble, scales, grains, crystals, '
    "noise, smooth, woven, organic].\n"
    '  "grain": dominant line/seam direction in the masked region, one of '
    "[horizontal, vertical, diagonal, none].\n"
    '  "relief": overall bumpiness, one of [flat, low, medium, high].\n'
    '  "edges": whether this region boundary or internal seams appear recessed, raised, '
    "or neither -- one of [recessed, raised, none].\n"
    '  "features": array (<=6) of obvious localized points, each '
    '{"type":..., "x":int, "y":int}; type in [rivet, stud, bump, gem, knot, crack, '
    "hole, scratch, highlight]. Use [] if unsure.\n"
    "ALLOWED materials: " + ", ".join(MATERIAL_NAMES) + "\n"
    'Example: {"palette":{"0":"cloth","1":"cloth"},"material":"cloth",'
    '"pattern":"woven","grain":"horizontal","relief":"low","edges":"none",'
    '"features":[]}'
)


def material_labels_dict():
    labels = {"0": "invalid"}
    labels.update({str(i): name for i, name in INDEX_TO_MATERIAL.items()})
    return labels


def normalize_material_name(name):
    mat = str(name).strip().lower()
    return mat if mat in MATERIALS else "unknown"


def encode_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def palette_text(palette, coverage):
    return "; ".join(
        f"#{i} rgb({r},{g},{b}) {coverage[i] * 100:.1f}%"
        for i, (r, g, b) in enumerate(palette)
    )


def extract_palette_masked(img_rgb, mask, ncolors):
    """Quantize only valid pixels, then assign every valid source pixel to a cluster."""
    rgb = np.array(img_rgb.convert("RGB"), dtype=np.uint8)
    valid = mask.astype(bool)
    samples = rgb[valid]
    if samples.size == 0:
        raise ValueError("masked palette has no valid pixels")

    max_samples = 262144
    if len(samples) > max_samples:
        step = int(np.ceil(len(samples) / max_samples))
        q_samples = samples[::step][:max_samples]
    else:
        q_samples = samples
    unique = np.unique(q_samples.reshape(-1, 3), axis=0)
    colors = min(max(1, int(ncolors)), max(1, len(unique)))
    side = int(np.ceil(np.sqrt(len(q_samples))))
    rows = int(np.ceil(len(q_samples) / side))
    sample_img = np.zeros((rows, side, 3), np.uint8)
    sample_img.reshape(-1, 3)[: len(q_samples)] = q_samples
    p = Image.fromarray(sample_img).convert(
        "P", palette=Image.ADAPTIVE, colors=colors, dither=Image.Dither.NONE
    )
    pal = p.getpalette()[: 256 * 3]
    used = np.unique(np.array(p, dtype=np.uint8).reshape(-1)[: len(q_samples)])
    palette = [(pal[u * 3], pal[u * 3 + 1], pal[u * 3 + 2]) for u in used]

    idx = np.zeros(valid.shape, np.int32)
    best = np.full(valid.shape, np.inf, np.float32)
    rgb_f = rgb.astype(np.float32)
    for i, c in enumerate(palette):
        col = np.array(c, np.float32)
        dist = np.sum((rgb_f - col) ** 2, axis=2)
        better = valid & (dist < best)
        idx[better] = i
        best[better] = dist[better]

    counts = np.bincount(idx[valid].reshape(-1), minlength=len(palette))
    used_order = [u for u in np.argsort(-counts) if counts[u] > 0]
    remap = {old: new for new, old in enumerate(used_order)}
    idx_remap = np.zeros_like(idx)
    for old, new in remap.items():
        idx_remap[(idx == old) & valid] = new
    palette = [palette[u] for u in used_order]
    coverage = [counts[u] / max(1, valid.sum()) for u in used_order]
    return idx_remap, palette, coverage


def _checker(size):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    a = ((xx // 16 + yy // 16) % 2).astype(np.uint8)
    lo = np.array((38, 38, 42), np.uint8)
    hi = np.array((62, 62, 68), np.uint8)
    return Image.fromarray(np.where(a[..., None] == 0, lo, hi))


def build_surface_prompt_image(img_rgb, palette, coverage, mask=None, max_preview=512):
    rgb = img_rgb.convert("RGB")
    if mask is not None:
        m = Image.fromarray(mask.astype(np.uint8) * 255).resize(rgb.size, Image.NEAREST)
        base = _checker(rgb.size)
        base.paste(rgb, mask=m)
        rgb = base

    w, h = rgb.size
    scale = min(max_preview / max(w, h), 1.0)
    preview = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    cols = min(len(palette), 6)
    rows = (len(palette) + cols - 1) // cols
    canvas_w = max(preview.width, cols * SWATCH)
    canvas_h = preview.height + rows * SWATCH + 8
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    canvas.paste(preview, ((canvas_w - preview.width) // 2, 0))
    d = ImageDraw.Draw(canvas)
    for i, c in enumerate(palette):
        cx, cy = (i % cols) * SWATCH, preview.height + 8 + (i // cols) * SWATCH
        d.rectangle([cx, cy, cx + SWATCH - 2, cy + SWATCH - 2], fill=c, outline=(255, 255, 255))
        lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        fg = (0, 0, 0) if lum > 110 else (255, 255, 255)
        d.text((cx + 3, cy + 2), f"{i}", fill=fg)
        d.text((cx + 3, cy + 14), f"{coverage[i] * 100:.0f}%", fill=fg)
    return canvas


def normalize_surface_result(raw, palette, width, height):
    pal_in = raw.get("palette", {}) or {}
    palette_map = {}
    for i in range(len(palette)):
        name = str(pal_in.get(str(i), "unknown")).strip().lower()
        palette_map[i] = name if name in MATERIALS else "unknown"

    feats = []
    for f in (raw.get("features") or [])[:6]:
        try:
            ftype = str(f.get("type", "")).lower()
            if ftype not in RAISED_FEATURES and ftype not in SUNKEN_FEATURES:
                continue
            x = f.get("x", f.get("col"))
            y = f.get("y", f.get("row"))
            if x is None or y is None:
                continue
            x, y = float(x), float(y)
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                x, y = x * (width - 1), y * (height - 1)
            elif max(x, y) <= 64:
                x, y = x * width / 32.0, y * height / 32.0
            feats.append({"type": ftype, "x": int(round(x)), "y": int(round(y))})
        except (ValueError, TypeError):
            pass

    material = str(raw.get("material", "unknown")).lower()
    if material not in MATERIALS:
        material = "unknown"

    return {
        "palette": palette_map,
        "material": material,
        "pattern": str(raw.get("pattern", "noise")).lower(),
        "grain": str(raw.get("grain", "none")).lower(),
        "relief": raw.get("relief", "medium") if raw.get("relief") in RELIEF_SCALE else "medium",
        "edges": raw.get("edges", "none") if raw.get("edges") in ("recessed", "raised", "none") else "none",
        "features": feats,
    }


def classify_surface(img_rgb, palette, coverage, name_hint, api_url, model, headers,
                     mask=None, timeout=120, max_tokens=1600, region=False):
    prompt_img = build_surface_prompt_image(img_rgb, palette, coverage, mask=mask)
    system_prompt = REGION_SYSTEM_PROMPT if region else SURFACE_SYSTEM_PROMPT
    subject = "masked region" if region else "texture"
    user_txt = (
        f"Filename hint: '{name_hint}'.\n"
        f"{subject.title()} size: {img_rgb.width}x{img_rgb.height}. "
        f"Palette ({len(palette)} colors, index rgb coverage): {palette_text(palette, coverage)}\n"
        "Return exactly one JSON object in message.content. No markdown, no wrapper tags, no prose."
    )
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\nThe final model message.content must contain the JSON object itself."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_txt},
                    {"type": "image_url", "image_url": {"url": encode_data_url(prompt_img)}},
                ],
            },
        ],
    }
    r = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    try:
        raw = extract_json_object(msg.get("content", ""))
    except ValueError:
        raw = extract_json_object(msg.get("reasoning_content", ""))
    return normalize_surface_result(raw, palette, img_rgb.width, img_rgb.height)


def _shift(a, dy, dx, wrap=False):
    if wrap:
        return np.roll(np.roll(a, dy, axis=0), dx, axis=1)

    out = np.zeros_like(a)
    h, w = a.shape[:2]
    y_src0, y_src1 = max(0, -dy), min(h, h - dy)
    x_src0, x_src1 = max(0, -dx), min(w, w - dx)
    y_dst0, y_dst1 = max(0, dy), min(h, h + dy)
    x_dst0, x_dst1 = max(0, dx), min(w, w + dx)
    out[y_dst0:y_dst1, x_dst0:x_dst1] = a[y_src0:y_src1, x_src0:x_src1]
    return out


def _box_blur(a, iterations=1, mask=None, wrap=False):
    out = a.astype(np.float32, copy=True)
    valid = np.ones(a.shape, np.float32) if mask is None else mask.astype(np.float32)
    for _ in range(iterations):
        acc = np.zeros_like(out, np.float32)
        weight = np.zeros_like(out, np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                v = _shift(valid, dy, dx, wrap)
                acc += _shift(out, dy, dx, wrap) * v
                weight += v
        out = np.divide(acc, np.maximum(weight, 1e-6))
        if mask is not None:
            out = np.where(mask, out, a)
    return out


def _boundary_strength(idx_map, mask=None, wrap=False):
    h, w = idx_map.shape
    valid = np.ones((h, w), bool) if mask is None else mask.astype(bool)
    total = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nidx = _shift(idx_map, dy, dx, wrap)
        nvalid = _shift(valid, dy, dx, wrap)
        both = valid & nvalid
        total += ((idx_map != nidx) & both).astype(np.float32)
        count += both.astype(np.float32)
    return np.divide(total, np.maximum(count, 1.0))


def _gradient(height, mask=None, wrap=False):
    valid = np.ones(height.shape, bool) if mask is None else mask.astype(bool)
    left = _shift(height, 0, 1, wrap)
    right = _shift(height, 0, -1, wrap)
    up = _shift(height, 1, 0, wrap)
    down = _shift(height, -1, 0, wrap)

    if not wrap:
        left_valid = _shift(valid, 0, 1, wrap)
        right_valid = _shift(valid, 0, -1, wrap)
        up_valid = _shift(valid, 1, 0, wrap)
        down_valid = _shift(valid, -1, 0, wrap)
        left = np.where(left_valid, left, height)
        right = np.where(right_valid, right, height)
        up = np.where(up_valid, up, height)
        down = np.where(down_valid, down, height)

    return (right - left) * 0.5, (down - up) * 0.5


def add_surface_dab(height, y, x, amp, sigma=2.5):
    h, w = height.shape
    rad = max(2, int(round(sigma * 2.5)))
    y0, y1 = max(0, y - rad), min(h, y + rad + 1)
    x0, x1 = max(0, x - rad), min(w, x + rad + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    weight = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma))
    height[y0:y1, x0:x1] += amp * weight


def _fill_invalid_with_mean(a, mask):
    if mask is None:
        return a
    if not mask.any():
        return a
    mean = float(np.mean(a[mask]))
    return np.where(mask, a, mean)


def build_surface_pbr(img_rgb, idx_map, res, gstrength, directx, invert_height,
                      mask=None, seamless=False, shadow_suppression=0.25):
    rgb = np.array(img_rgb.convert("RGB"), dtype=np.float32)
    h, w = idx_map.shape
    valid = np.ones((h, w), bool) if mask is None else mask.astype(bool)
    shadow_suppression = float(np.clip(shadow_suppression, 0.0, 1.0))
    assignment = res["palette"]
    relief = RELIEF_SCALE.get(res["relief"], 1.0)

    rough = np.full((h, w), MATERIALS["unknown"]["rough"], np.float32)
    metal = np.full((h, w), MATERIALS["unknown"]["metal"], np.float32)
    nstr = np.full((h, w), MATERIALS["unknown"]["nstr"], np.float32)
    aost = np.full((h, w), MATERIALS["unknown"]["ao"], np.float32)
    for i, mat in assignment.items():
        m = MATERIALS.get(mat, MATERIALS["unknown"])
        msk = (idx_map == i) & valid
        rough[msk], metal[msk] = m["rough"], m["metal"]
        nstr[msk], aost[msk] = m["nstr"], m["ao"]

    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
    lum = _fill_invalid_with_mean(lum, valid)
    if shadow_suppression > 0:
        low = _box_blur(lum, iterations=10, mask=valid, wrap=seamless)
        high = (lum - low) * (1.0 - shadow_suppression)
        broad = lum * shadow_suppression
        bias = 0.5 * (1.0 - shadow_suppression)
        lum = np.clip(high + broad + bias, 0, 1)
    height = (1.0 - lum) if invert_height else lum
    height = 0.5 + (height - 0.5) * 0.55

    if res["edges"] != "none":
        bs = _box_blur(_boundary_strength(idx_map, valid, seamless), iterations=1, mask=valid, wrap=seamless)
        depth = 0.28 * relief * (1 if res["edges"] == "raised" else -1)
        height = np.clip(height + bs * depth, 0, 1)

    sigma = max(1.5, min(h, w) / 160.0)
    for f in res["features"]:
        amp = 0.22 * relief
        if f["type"] in SUNKEN_FEATURES:
            amp = -amp
        elif f["type"] not in RAISED_FEATURES:
            amp *= 0.4
        x = int(np.clip(f.get("x", 0), 0, w - 1))
        y = int(np.clip(f.get("y", 0), 0, h - 1))
        add_surface_dab(height, y, x, amp, sigma=sigma)
    height = np.clip(np.where(valid, height, 0.0), 0, 1)

    gx, gy = _gradient(height, valid, seamless)
    if res["grain"] == "horizontal":
        gx *= 0.8
        gy *= 1.3
    elif res["grain"] == "vertical":
        gx *= 1.3
        gy *= 0.8
    strength = nstr * gstrength * relief
    nx = -gx * strength
    ny = (gy if directx else -gy) * strength
    nz = np.ones_like(height)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = (np.stack([nx * inv, ny * inv, nz * inv], -1) * 0.5 + 0.5)
    normal[~valid] = (0.5, 0.5, 1.0)

    blurred = _box_blur(height, iterations=2, mask=valid, wrap=seamless)
    ao = np.clip(1.0 - np.clip((blurred - height) * 2.0 * aost, 0, 1), 0, 1)
    ao[~valid] = 1.0
    rough[~valid] = MATERIALS["unknown"]["rough"]
    metal[~valid] = 0.0

    height_png = (height * 255).clip(0, 255).astype(np.uint8)
    normal_png = (normal * 255).clip(0, 255).astype(np.uint8)
    orm = (np.stack([ao, rough, metal], -1) * 255).clip(0, 255).astype(np.uint8)
    return normal_png, height_png, orm


def dilate_array(arr, mask, iterations, target_mask=None):
    if mask is None or iterations <= 0:
        return arr
    valid = mask.astype(bool).copy()
    target = np.ones_like(valid, bool) if target_mask is None else target_mask.astype(bool).copy()
    target |= valid
    valid &= target
    out = arr.astype(np.float32, copy=True)
    if arr.ndim == 2:
        work = out[..., None]
    else:
        work = out

    for _ in range(iterations):
        fill = target & ~valid
        if not fill.any():
            break
        acc = np.zeros_like(work, np.float32)
        weight = np.zeros(valid.shape, np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nvalid = _shift(valid, dy, dx, False)
                acc += _shift(work, dy, dx, False) * nvalid[..., None]
                weight += nvalid.astype(np.float32)
        grow = fill & (weight > 0)
        if not grow.any():
            break
        work[grow] = acc[grow] / weight[grow, None]
        valid[grow] = True

    if arr.ndim == 2:
        return work[..., 0].clip(0, 255).astype(arr.dtype)
    return work.clip(0, 255).astype(arr.dtype)


def dilate_mask(mask, iterations, target_mask=None):
    if mask is None or iterations <= 0:
        return mask
    out = mask.astype(bool).copy()
    target = np.ones_like(out, bool) if target_mask is None else target_mask.astype(bool).copy()
    target |= out
    out &= target
    for _ in range(iterations):
        fill = target & ~out
        if not fill.any():
            break
        grow = np.zeros_like(out, bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                grow |= _shift(out, dy, dx, False)
        out |= fill & grow
    return out


def _guide_image_for_resistance(arr, mask, iterations, target_mask=None):
    guide = arr
    if mask is not None and iterations > 0:
        guide = dilate_array(arr, mask, iterations, target_mask=target_mask)
    guide = guide.astype(np.float32) / 255.0
    if guide.ndim == 2:
        guide = guide[..., None]
    return guide


def resistance_dilate_array(arr, mask, iterations, resistance=10.0, target_mask=None):
    """Edge-aware UV-space dilation.

    Missing texels are filled from neighboring valid texels, but neighbors separated
    by large color changes in a provisional guide image receive lower weight.
    """
    if mask is None or iterations <= 0:
        return arr
    valid = mask.astype(bool).copy()
    target = np.ones_like(valid, bool) if target_mask is None else target_mask.astype(bool).copy()
    target |= valid
    valid &= target
    out = arr.astype(np.float32, copy=True)
    work = out[..., None] if arr.ndim == 2 else out
    guide = _guide_image_for_resistance(arr, mask, iterations, target_mask=target)
    resistance = max(0.0, float(resistance))

    for _ in range(iterations):
        fill = target & ~valid
        if not fill.any():
            break
        acc = np.zeros_like(work, np.float32)
        weight = np.zeros(valid.shape, np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nvalid = _shift(valid, dy, dx, False)
                if not nvalid.any():
                    continue
                nguide = _shift(guide, dy, dx, False)
                diff = np.sqrt(np.mean((guide - nguide) ** 2, axis=-1))
                w = nvalid.astype(np.float32) * np.exp(-resistance * diff)
                acc += _shift(work, dy, dx, False) * w[..., None]
                weight += w
        grow = fill & (weight > 1e-6)
        if not grow.any():
            break
        work[grow] = acc[grow] / weight[grow, None]
        valid[grow] = True

    if arr.ndim == 2:
        return work[..., 0].clip(0, 255).astype(arr.dtype)
    return work.clip(0, 255).astype(arr.dtype)


def resistance_dilate_labels(labels, source_mask, iterations, guide, resistance=10.0, target_mask=None):
    """Fill integer material IDs through low-resistance UV-space neighborhoods."""
    if iterations <= 0:
        return labels
    out = labels.astype(np.uint8, copy=True)
    valid = source_mask.astype(bool) & (out > 0)
    if not valid.any():
        return out
    target = np.ones_like(valid, bool) if target_mask is None else target_mask.astype(bool)
    guide_img = _guide_image_for_resistance(guide, target, max(1, iterations), target_mask=target)
    resistance = max(0.0, float(resistance))

    for _ in range(iterations):
        fill = target & ~valid
        if not fill.any():
            break
        present = [int(v) for v in np.unique(out[valid]) if int(v) > 0]
        if not present:
            break
        scores = {mid: np.zeros(valid.shape, np.float32) for mid in present}
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nvalid = _shift(valid, dy, dx, False)
                if not nvalid.any():
                    continue
                nlabels = _shift(out, dy, dx, False)
                nguide = _shift(guide_img, dy, dx, False)
                diff = np.sqrt(np.mean((guide_img - nguide) ** 2, axis=-1))
                w = nvalid.astype(np.float32) * np.exp(-resistance * diff)
                for mid in present:
                    scores[mid] += w * (nlabels == mid)
        best_score = np.zeros(valid.shape, np.float32)
        best_label = np.zeros(valid.shape, np.uint8)
        for mid, score in scores.items():
            better = score > best_score
            best_score[better] = score[better]
            best_label[better] = mid
        grow = fill & (best_score > 1e-6)
        if not grow.any():
            break
        out[grow] = best_label[grow]
        valid[grow] = True
    return out


def load_mask(mask_path, size, alpha_threshold=1):
    if mask_path is None:
        return None
    m = Image.open(mask_path)
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    if "A" in m.getbands():
        arr = np.array(m.convert("RGBA"))[..., 3]
    else:
        arr = np.array(m.convert("L"))
    return arr > alpha_threshold


def default_material_labels_path(material_map_path):
    p = Path(material_map_path)
    candidate = p.with_suffix(".json")
    return candidate if candidate.exists() else None


def load_material_labels(labels_path=None):
    labels = material_labels_dict()
    if labels_path:
        raw = json.loads(Path(labels_path).read_text())
        for key, value in raw.items():
            labels[str(int(key))] = normalize_material_name(value)
    return labels


def load_material_map(material_map_path, size, labels_path=None):
    m = Image.open(material_map_path)
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    arr = np.array(m.convert("L"), dtype=np.int32)
    labels = load_material_labels(labels_path or default_material_labels_path(material_map_path))
    return arr, labels


def result_from_material_map(material_map, labels, mask=None):
    valid = material_map > 0
    if mask is not None:
        valid &= mask.astype(bool)
    ids, counts = np.unique(material_map[valid], return_counts=True)
    palette = {}
    for raw_id in ids:
        mat = normalize_material_name(labels.get(str(int(raw_id)), "unknown"))
        palette[int(raw_id)] = mat
    if len(ids) == 0:
        palette[0] = "unknown"
        dominant = "unknown"
    else:
        dominant_id = int(ids[int(np.argmax(counts))])
        dominant = palette.get(dominant_id, "unknown")
    return {
        "palette": palette,
        "material": dominant,
        "pattern": "noise",
        "grain": "none",
        "relief": "medium",
        "edges": "recessed",
        "features": [],
    }


def load_texture(path, mask_path=None, use_alpha=True, alpha_threshold=1):
    im = Image.open(path)
    mask = None
    if use_alpha and ("A" in im.getbands() or "transparency" in im.info):
        rgba = im.convert("RGBA")
        mask = np.array(rgba)[..., 3] > alpha_threshold
        rgb = rgba.convert("RGB")
    else:
        rgb = im.convert("RGB")
    explicit = load_mask(mask_path, rgb.size, alpha_threshold=alpha_threshold)
    if explicit is not None:
        mask = explicit if mask is None else (mask & explicit)
    return rgb, mask


def resolve_mask_path(mask_arg, input_path, input_root, input_is_file):
    if not mask_arg:
        return None
    root = Path(mask_arg)
    if root.is_file():
        if not input_is_file:
            raise SystemExit("--mask as a file is only valid when the input is one file")
        return root
    if not root.is_dir():
        raise SystemExit(f"mask path not found: {root}")
    rel = input_path.relative_to(input_root)
    candidates = [
        root / rel,
        root / rel.with_name(rel.stem + "_mask" + rel.suffix),
        root / rel.with_suffix(".png"),
        root / rel.with_name(rel.stem + "_mask.png"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_material_map_path(material_map_arg, input_path, input_root, input_is_file):
    if not material_map_arg:
        return None
    root = Path(material_map_arg)
    if root.is_file():
        if not input_is_file:
            raise SystemExit("--material-map as a file is only valid when the input is one file")
        return root
    if not root.is_dir():
        raise SystemExit(f"material-map path not found: {root}")
    rel = input_path.relative_to(input_root)
    candidates = [
        root / rel,
        root / rel.with_name(rel.stem + "_materials" + rel.suffix),
        root / rel.with_suffix(".png"),
        root / rel.with_name(rel.stem + "_materials.png"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def output_stem(out_dir, p, input_root, input_is_file):
    if input_is_file:
        stem = out_dir / p.stem
    else:
        stem = out_dir / p.relative_to(input_root).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    return stem


def store_manifest(manifest, man_path, key, h, res):
    store = copy_res(res)
    store["palette"] = {str(k): v for k, v in res["palette"].items()}
    manifest[key] = {"hash": h, "result": store}
    tmp = man_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, man_path)


def collect_input_files(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p], p.parent, True
    if not p.is_dir():
        raise SystemExit(f"input not found: {p}")
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
    return files, p, False


def combined_hash(image_path, mask_path=None, extra_paths=None):
    h = file_hash(image_path)
    if mask_path is not None:
        h += ":" + file_hash(mask_path)
    for extra in extra_paths or []:
        if extra:
            h += ":" + file_hash(Path(extra))
    return h


def atlas_command(args):
    files, input_root, input_is_file = collect_input_files(args.input)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No images found in {args.input}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "_surface_materials_manifest.json"
    manifest = load_manifest(man_path)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    model = "(heuristic)" if args.no_ai else None
    if model is None and not args.material_map:
        model = resolve_model(args.model, args.api_url, headers)
    print(f"{len(files)} surface textures -> {out_dir}  (model: {model or '(lazy)'})")

    ai_calls = ai_fail = dedup_hits = skipped = 0
    content_cache = {}
    for p in tqdm(files, unit="tex"):
        try:
            mask_path = resolve_mask_path(args.mask, p, input_root, input_is_file)
            material_map_path = resolve_material_map_path(args.material_map, p, input_root, input_is_file)
            labels_path = Path(args.material_labels) if args.material_labels else None
            img, mask = load_texture(
                p,
                mask_path=mask_path,
                use_alpha=not args.ignore_alpha,
                alpha_threshold=args.alpha_threshold,
            )
            stem = output_stem(out_dir, p, input_root, input_is_file)
            outs = (f"{stem}_n.png", f"{stem}_h.png", f"{stem}_orm.png")
            if args.write_mask and mask is not None:
                outs += (f"{stem}_mask.png",)
            if not args.force and all(os.path.exists(o) for o in outs):
                skipped += 1
                continue

            extra_hashes = [material_map_path]
            if labels_path is not None and labels_path.exists():
                extra_hashes.append(labels_path)
            h = combined_hash(p, mask_path, extra_hashes)
            key = str(p.relative_to(input_root) if not input_is_file else Path(p.name)).replace("\\", "/")

            if material_map_path is not None:
                idx_map, material_labels = load_material_map(material_map_path, img.size, labels_path)
                res = result_from_material_map(idx_map, material_labels, mask=mask)
                store_manifest(manifest, man_path, key, h, res)
            elif not args.force and manifest.get(key, {}).get("hash") == h:
                res = manifest[key]["result"]
                res["palette"] = {int(k): v for k, v in res["palette"].items()}
            else:
                idx_map, palette, coverage = extract_palette(img, args.palette_colors)
                cache_parts = [img.tobytes()]
                if mask is not None:
                    cache_parts.append(mask.astype(np.uint8).tobytes())
                chash = hashlib.md5(b"".join(cache_parts)).hexdigest()[:16]
                if args.no_ai:
                    raw = heuristic_result(palette)
                elif chash in content_cache:
                    raw = content_cache[chash]
                    dedup_hits += 1
                else:
                    if model is None:
                        model = resolve_model(args.model, args.api_url, headers)
                    try:
                        raw = classify_surface(
                            img,
                            palette,
                            coverage,
                            clean_name_hint(p.stem),
                            args.api_url,
                            model,
                            headers,
                            mask=mask,
                            max_tokens=args.vision_max_tokens,
                        )
                    except Exception as e:
                        ai_fail += 1
                        tqdm.write(f"  AI fail {p.name}: {e} -> heuristic")
                        raw = heuristic_result(palette)
                    content_cache[chash] = raw
                    ai_calls += 1
                    if args.reset_every and ai_calls % args.reset_every == 0:
                        reset_context(args.api_url, model, headers, args.reset_cmd)

                res = copy_res(raw)
                if not args.no_filename_prior:
                    res = apply_filename_prior(res, p.stem)
                store_manifest(manifest, man_path, key, h, res)

            normal, height, orm = build_surface_pbr(
                img,
                idx_map,
                res,
                args.strength,
                args.dx,
                args.invert_height,
                mask=mask,
                seamless=args.seamless,
                shadow_suppression=args.shadow_suppression,
            )
            if mask is not None and args.dilate_padding:
                normal = dilate_array(normal, mask, args.dilate_padding)
                height = dilate_array(height, mask, args.dilate_padding)
                orm = dilate_array(orm, mask, args.dilate_padding)

            Image.fromarray(normal).save(f"{stem}_n.png")
            Image.fromarray(height).save(f"{stem}_h.png")
            Image.fromarray(orm).save(f"{stem}_orm.png")
            if args.write_mask and mask is not None:
                Image.fromarray(mask.astype(np.uint8) * 255).save(f"{stem}_mask.png")
        except Exception as e:
            tqdm.write(f"  ERROR {p.name}: {e}")

    print(
        f"Done. {len(files)} files: {skipped} already done, "
        f"{ai_calls} AI calls, {dedup_hits} dedup hits, {ai_fail} fallbacks."
    )
    print(f"Manifest (editable): {man_path}")


def parse_size(s):
    if "x" in s.lower():
        w, h = re.split("[xX]", s)
        return int(w), int(h)
    n = int(s)
    return n, n


def find_capture_pairs(folder, color_suffix, uv_suffix):
    root = Path(folder)
    if not root.is_dir():
        raise SystemExit(f"capture folder not found: {root}")
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    by_key = {}
    for p in images:
        stem = p.stem
        if stem.endswith(color_suffix):
            key = str(p.with_name(stem[: -len(color_suffix)]).relative_to(root)).replace("\\", "/")
            by_key.setdefault(key, {})["color"] = p
        elif stem.endswith(uv_suffix):
            key = str(p.with_name(stem[: -len(uv_suffix)]).relative_to(root)).replace("\\", "/")
            by_key.setdefault(key, {})["uv"] = p
    return [(v["color"], v["uv"]) for _, v in sorted(by_key.items()) if "color" in v and "uv" in v]


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_to_mask(img_rgb, mask, pad=16):
    bbox = mask_bbox(mask)
    if bbox is None:
        raise ValueError("cannot crop empty mask")
    x0, y0, x1, y1 = bbox
    w, h = img_rgb.size
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    return img_rgb.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1], (x0, y0, x1, y1)


def capture_key_path(color_path, captures_root, color_suffix):
    rel = Path(color_path).relative_to(Path(captures_root))
    stem = rel.stem
    if stem.endswith(color_suffix):
        stem = stem[: -len(color_suffix)]
    return rel.with_name(stem)


def component_regions(idx_map, valid, min_area, max_regions):
    """Split a label map into connected region masks, largest first."""
    h, w = idx_map.shape
    visited = np.zeros((h, w), bool)
    regions = []
    ys, xs = np.nonzero(valid)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx] or not valid[sy, sx]:
            continue
        label = int(idx_map[sy, sx])
        q = deque([(int(sy), int(sx))])
        visited[sy, sx] = True
        cy, cx = [], []
        while q:
            y0, x0 = q.popleft()
            cy.append(y0)
            cx.append(x0)
            for ny, nx in ((y0 - 1, x0), (y0 + 1, x0), (y0, x0 - 1), (y0, x0 + 1)):
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    if valid[ny, nx] and int(idx_map[ny, nx]) == label:
                        visited[ny, nx] = True
                        q.append((ny, nx))
        area = len(cy)
        if area < min_area:
            continue
        m = np.zeros((h, w), bool)
        m[np.array(cy, dtype=np.int32), np.array(cx, dtype=np.int32)] = True
        bbox = mask_bbox(m)
        regions.append({
            "id": len(regions),
            "mask": m,
            "area": area,
            "bbox": bbox,
            "source": "connected",
            "cluster": label,
        })

    regions.sort(key=lambda r: r["area"], reverse=True)
    if max_regions > 0:
        regions = regions[:max_regions]
    for i, region in enumerate(regions):
        region["id"] = i
    return regions


def connected_segment_regions(color, valid, args):
    idx_screen, _, _ = extract_palette_masked(color, valid, args.segment_palette_colors)
    min_area = max(args.segment_min_area, int(valid.sum() * args.segment_min_coverage))
    return component_regions(idx_screen, valid, min_area, args.segment_max_regions)


def _mask_from_file(path, size, threshold):
    m = Image.open(path)
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    rgba = np.array(m.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3] > threshold
    luma = (
        0.299 * rgba[..., 0].astype(np.float32)
        + 0.587 * rgba[..., 1].astype(np.float32)
        + 0.114 * rgba[..., 2].astype(np.float32)
    ) > threshold
    if alpha.any() and not alpha.all():
        return alpha & (luma | (rgba[..., 3] > threshold))
    return luma


def external_segment_regions(color_path, color, valid, args):
    root = Path(args.segment_masks)
    if not root.is_dir():
        raise ValueError(f"segment mask folder not found: {root}")
    key = capture_key_path(color_path, args.captures, args.color_suffix)
    parent = root / key.parent
    files = []
    mask_dir = root / key
    if mask_dir.is_dir():
        files.extend(p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if parent.is_dir():
        for pattern in (
            f"{key.name}_mask*",
            f"{key.name}-mask*",
            f"{key.name}_seg*",
            f"{key.name}-seg*",
        ):
            files.extend(p for p in parent.glob(pattern) if p.suffix.lower() in IMAGE_EXTS)
    flat_prefix = str(key).replace("\\", "__").replace("/", "__")
    for pattern in (f"{flat_prefix}_mask*", f"{flat_prefix}_seg*"):
        files.extend(p for p in root.glob(pattern) if p.suffix.lower() in IMAGE_EXTS)

    unique = []
    seen = set()
    for p in sorted(files):
        rp = str(p.resolve()).lower()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    min_area = max(args.segment_min_area, int(valid.sum() * args.segment_min_coverage))
    regions = []
    for p in unique:
        m = _mask_from_file(p, color.size, args.alpha_threshold) & valid
        area = int(m.sum())
        if area < min_area:
            continue
        regions.append({
            "id": len(regions),
            "mask": m,
            "area": area,
            "bbox": mask_bbox(m),
            "source": "masks",
            "file": str(p),
        })
    regions.sort(key=lambda r: r["area"], reverse=True)
    if args.segment_max_regions > 0:
        regions = regions[:args.segment_max_regions]
    for i, region in enumerate(regions):
        region["id"] = i
    return regions


def sam_segment_regions(color, valid, args):
    if args.segmenter == "sam2":
        try:
            import torch
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except Exception as e:
            raise RuntimeError("SAM2 backend requires the sam2 package and torch") from e
        if not args.sam2_checkpoint or not args.sam2_config:
            raise ValueError("--segmenter sam2 requires --sam2-checkpoint and --sam2-config")
        device = args.sam_device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
        generator = SAM2AutomaticMaskGenerator(
            sam,
            points_per_side=args.sam_points_per_side,
            pred_iou_thresh=args.sam_pred_iou_thresh,
            stability_score_thresh=args.sam_stability_score_thresh,
        )
    else:
        try:
            import torch
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except Exception as e:
            raise RuntimeError("SAM backend requires the segment-anything package and torch") from e
        if not args.sam_checkpoint:
            raise ValueError("--segmenter sam requires --sam-checkpoint")
        device = args.sam_device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
        sam.to(device=device)
        generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=args.sam_points_per_side,
            pred_iou_thresh=args.sam_pred_iou_thresh,
            stability_score_thresh=args.sam_stability_score_thresh,
        )

    anns = generator.generate(np.array(color.convert("RGB")))
    min_area = max(args.segment_min_area, int(valid.sum() * args.segment_min_coverage))
    regions = []
    for ann in anns:
        m = np.array(ann["segmentation"], dtype=bool) & valid
        area = int(m.sum())
        if area < min_area:
            continue
        regions.append({
            "id": len(regions),
            "mask": m,
            "area": area,
            "bbox": mask_bbox(m),
            "source": args.segmenter,
            "score": float(ann.get("predicted_iou", ann.get("stability_score", 0.0))),
        })
    regions.sort(key=lambda r: (r.get("score", 0.0), r["area"]), reverse=True)
    if args.segment_max_regions > 0:
        regions = regions[:args.segment_max_regions]
    for i, region in enumerate(regions):
        region["id"] = i
    return regions


def segment_regions_for_capture(color_path, color, valid, args):
    if args.segmenter == "masks":
        regions = external_segment_regions(color_path, color, valid, args)
    elif args.segmenter in ("sam", "sam2"):
        regions = sam_segment_regions(color, valid, args)
    else:
        regions = connected_segment_regions(color, valid, args)
    if not regions:
        raise ValueError("segmenter produced no valid regions")
    return regions


def masked_region_hash(color, mask):
    crop, crop_mask, _ = crop_to_mask(color, mask, pad=0)
    rgba = crop.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8)
    arr[..., 3] = crop_mask.astype(np.uint8) * 255
    thumb = Image.fromarray(arr).resize((96, 96), Image.Resampling.LANCZOS)
    return hashlib.md5(np.array(thumb, dtype=np.uint8).tobytes()).hexdigest()[:16]


def dominant_material_from_result(res, palette, coverage):
    mat = normalize_material_name(res.get("material", "unknown"))
    if mat != "unknown":
        return mat
    weights = {}
    for i, cov in enumerate(coverage):
        pm = normalize_material_name(res.get("palette", {}).get(i, "unknown"))
        weights[pm] = weights.get(pm, 0.0) + float(cov)
    if not weights:
        return "unknown"
    return max(weights.items(), key=lambda kv: kv[1])[0]


def summarize_screen_materials(screen_materials, valid):
    res = result_from_material_map(screen_materials.astype(np.int32), material_labels_dict(), mask=valid)
    store = copy_res(res)
    store["palette"] = {str(k): v for k, v in res["palette"].items()}
    return store


def write_segment_debug(args, color_path, regions, screen_materials):
    if not args.segments_output:
        return
    root = Path(args.segments_output)
    key = capture_key_path(color_path, args.captures, args.color_suffix)
    out_base = root / key.parent
    out_base.mkdir(parents=True, exist_ok=True)
    h, w = screen_materials.shape
    seg_vis = np.zeros((h, w, 3), np.uint8)
    for region in regions:
        rid = region["id"] + 1
        color_val = np.array([
            (rid * 73) % 255,
            (rid * 151) % 255,
            (rid * 211) % 255,
        ], dtype=np.uint8)
        seg_vis[region["mask"]] = color_val
    Image.fromarray(seg_vis).save(out_base / f"{key.name}_segments.png")
    Image.fromarray(screen_materials.astype(np.uint8)).save(out_base / f"{key.name}_materials_screen.png")


def classify_segment_regions(color, valid, regions, color_path, args, model, headers,
                             capture_cache, counters):
    screen_materials = np.zeros(valid.shape, np.uint8)
    screen_score = np.zeros(valid.shape, np.float32)
    segment_manifest = []
    for region in regions:
        rmask = region["mask"] & valid
        if not rmask.any():
            continue
        try:
            crop, crop_mask, bbox = crop_to_mask(color, rmask, pad=args.segment_crop_pad)
            _, palette, coverage = extract_palette_masked(crop, crop_mask, args.segment_region_colors)
            chash = masked_region_hash(color, rmask)
            if args.no_ai:
                raw = heuristic_result(palette)
            elif chash in capture_cache:
                raw = capture_cache[chash]
                counters["dedup_hits"] += 1
            else:
                try:
                    raw = classify_surface(
                        crop,
                        palette,
                        coverage,
                        clean_name_hint(color_path.stem) + f" region {region['id']}",
                        args.api_url,
                        model,
                        headers,
                        mask=crop_mask,
                        max_tokens=args.vision_max_tokens,
                        region=True,
                    )
                except Exception as e:
                    counters["ai_fail"] += 1
                    tqdm.write(f"  AI fail {color_path.name} region {region['id']}: {e} -> heuristic")
                    raw = heuristic_result(palette)
                capture_cache[chash] = raw
                counters["ai_calls"] += 1
                if args.reset_every and counters["ai_calls"] % args.reset_every == 0:
                    reset_context(args.api_url, model, headers, args.reset_cmd)

            res = copy_res(raw)
            if not args.no_filename_prior:
                res = apply_filename_prior(res, color_path.stem)
            material = dominant_material_from_result(res, palette, coverage)
            material_id = MATERIAL_TO_INDEX.get(material, MATERIAL_TO_INDEX["unknown"])
            specificity = 1.0 - min(1.0, float(region["area"]) / max(1.0, float(valid.sum())))
            score = float(region.get("score", 1.0)) + specificity
            update = rmask & (score >= screen_score)
            screen_materials[update] = material_id
            screen_score[update] = score
            store = copy_res(res)
            store["palette"] = {str(k): v for k, v in res["palette"].items()}
            segment_manifest.append({
                "id": int(region["id"]),
                "source": region.get("source", "unknown"),
                "area": int(region["area"]),
                "coverage": float(region["area"] / max(1, int(valid.sum()))),
                "bbox": [int(v) for v in bbox],
                "material": material,
                "material_id": int(material_id),
                "result": store,
            })
        except Exception as e:
            tqdm.write(f"  segment fail {color_path.name} region {region.get('id', '?')}: {e}")

    if args.segment_fill_unassigned:
        unassigned = valid & (screen_materials == 0)
        if unassigned.any():
            idx_screen, palette, coverage = extract_palette_masked(color, unassigned, args.palette_colors)
            if args.no_ai:
                raw = heuristic_result(palette)
            else:
                try:
                    raw = classify_surface(
                        color,
                        palette,
                        coverage,
                        clean_name_hint(color_path.stem) + " unassigned",
                        args.api_url,
                        model,
                        headers,
                        mask=unassigned,
                        max_tokens=args.vision_max_tokens,
                        region=True,
                    )
                    counters["ai_calls"] += 1
                except Exception as e:
                    counters["ai_fail"] += 1
                    tqdm.write(f"  AI fail {color_path.name} unassigned: {e} -> heuristic")
                    raw = heuristic_result(palette)
            res = copy_res(raw)
            for cluster_id, mat in res["palette"].items():
                material_id = MATERIAL_TO_INDEX.get(normalize_material_name(mat), MATERIAL_TO_INDEX["unknown"])
                screen_materials[(idx_screen == int(cluster_id)) & unassigned] = material_id

    return screen_materials, segment_manifest


def bake_captures_command(args):
    if args.segment_materials:
        args.tag_materials = True
    if args.segmenter == "masks" and not args.segment_masks:
        raise SystemExit("--segmenter masks requires --segment-masks")

    pairs = find_capture_pairs(args.captures, args.color_suffix, args.uv_suffix)
    if not pairs:
        raise SystemExit(
            f"No capture pairs found. Expected '*{args.color_suffix}.png' and '*{args.uv_suffix}.png'."
        )

    out_size = parse_size(args.size)
    width, height = out_size
    accum = np.zeros((height, width, 3), np.float64)
    count = np.zeros((height, width), np.float64)
    material_votes = {}
    capture_manifest = {}
    capture_cache = {}
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    model = None
    counters = {"ai_calls": 0, "ai_fail": 0, "dedup_hits": 0}
    if args.tag_materials:
        model = "(heuristic)" if args.no_ai else resolve_model(args.model, args.api_url, headers)
        mode = "segmented" if args.segment_materials else "palette"
        print(f"{len(pairs)} capture pairs -> {mode} semantic material atlas  (model: {model})")

    for color_path, uv_path in tqdm(pairs, unit="pair"):
        color = Image.open(color_path).convert("RGB")
        uv_img = Image.open(uv_path).convert("RGBA")
        if color.size != uv_img.size:
            uv_img = uv_img.resize(color.size, Image.NEAREST)

        c = np.array(color, dtype=np.float64)
        uv = np.array(uv_img, dtype=np.uint8)
        valid = uv[..., 3] > args.alpha_threshold
        if args.black_invalid:
            valid &= (uv[..., 0].astype(np.int32) + uv[..., 1].astype(np.int32) + uv[..., 2].astype(np.int32)) > args.alpha_threshold

        if not valid.any():
            continue

        u = uv[..., 0].astype(np.float64) / 255.0
        v = uv[..., 1].astype(np.float64) / 255.0
        if args.flip_v:
            v = 1.0 - v
        x = np.clip(np.rint(u * (width - 1)).astype(np.int32), 0, width - 1)
        y = np.clip(np.rint(v * (height - 1)).astype(np.int32), 0, height - 1)
        yy = y[valid]
        xx = x[valid]
        for ch in range(3):
            np.add.at(accum[..., ch], (yy, xx), c[..., ch][valid])
        np.add.at(count, (yy, xx), 1.0)

        if args.tag_materials:
            try:
                segment_manifest = None
                if args.segment_materials:
                    regions = segment_regions_for_capture(color_path, color, valid, args)
                    screen_materials, segment_manifest = classify_segment_regions(
                        color,
                        valid,
                        regions,
                        color_path,
                        args,
                        model,
                        headers,
                        capture_cache,
                        counters,
                    )
                    write_segment_debug(args, color_path, regions, screen_materials)
                    res = summarize_screen_materials(screen_materials, valid)
                else:
                    idx_screen, palette, coverage = extract_palette_masked(color, valid, args.palette_colors)
                    chash = hashlib.md5(
                        np.array(color, dtype=np.uint8).tobytes() + valid.astype(np.uint8).tobytes()
                    ).hexdigest()[:16]
                    if args.no_ai:
                        raw = heuristic_result(palette)
                    elif chash in capture_cache:
                        raw = capture_cache[chash]
                        counters["dedup_hits"] += 1
                    else:
                        try:
                            raw = classify_surface(
                                color,
                                palette,
                                coverage,
                                clean_name_hint(color_path.stem),
                                args.api_url,
                                model,
                                headers,
                                mask=valid,
                                max_tokens=args.vision_max_tokens,
                            )
                        except Exception as e:
                            counters["ai_fail"] += 1
                            tqdm.write(f"  AI fail {color_path.name}: {e} -> heuristic")
                            raw = heuristic_result(palette)
                        capture_cache[chash] = raw
                        counters["ai_calls"] += 1
                        if args.reset_every and counters["ai_calls"] % args.reset_every == 0:
                            reset_context(args.api_url, model, headers, args.reset_cmd)

                    res = copy_res(raw)
                    if not args.no_filename_prior:
                        res = apply_filename_prior(res, color_path.stem)
                    screen_materials = np.zeros(valid.shape, np.uint8)
                    for cluster_id, mat in res["palette"].items():
                        material_id = MATERIAL_TO_INDEX.get(normalize_material_name(mat), MATERIAL_TO_INDEX["unknown"])
                        screen_materials[(idx_screen == int(cluster_id)) & valid] = material_id
                for material_id in np.unique(screen_materials[valid]):
                    if material_id == 0:
                        continue
                    votes = material_votes.setdefault(int(material_id), np.zeros((height, width), np.uint32))
                    material_mask = valid & (screen_materials == material_id)
                    np.add.at(votes, (y[material_mask], x[material_mask]), 1)

                key = str(color_path.relative_to(Path(args.captures))).replace("\\", "/")
                store = copy_res(res)
                store["palette"] = {str(k): v for k, v in res["palette"].items()}
                capture_manifest[key] = {
                    "hash": combined_hash(color_path, uv_path),
                    "uv": str(uv_path.relative_to(Path(args.captures))).replace("\\", "/"),
                    "result": store,
                }
                if segment_manifest is not None:
                    capture_manifest[key]["mode"] = "segments"
                    capture_manifest[key]["segmenter"] = args.segmenter
                    capture_manifest[key]["segments"] = segment_manifest
                    capture_manifest[key]["unassigned_valid_pixels"] = int(
                        np.count_nonzero(valid & (screen_materials == 0))
                    )
            except Exception as e:
                tqdm.write(f"  tag fail {color_path.name}: {e}")

    mask = count > 0
    if not mask.any():
        raise SystemExit("No valid UV samples were written")
    fill_target_mask = None
    if args.fill_target_mask:
        fill_target_mask = load_mask(args.fill_target_mask, (width, height), args.alpha_threshold)
    out = np.divide(accum, np.maximum(count[..., None], 1.0)).clip(0, 255).astype(np.uint8)
    filled_mask = mask.copy()
    if args.dilate:
        if args.dilate_mode == "resistance":
            out = resistance_dilate_array(
                out,
                mask,
                args.dilate,
                args.dilate_resistance,
                target_mask=fill_target_mask,
            )
        else:
            out = dilate_array(out, mask, args.dilate, target_mask=fill_target_mask)
        filled_mask = dilate_mask(mask, args.dilate, target_mask=fill_target_mask)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path)
    mask_path = args.mask_output or str(out_path.with_name(out_path.stem + "_mask.png"))
    output_mask = filled_mask if args.expand_mask else mask
    Image.fromarray(output_mask.astype(np.uint8) * 255).save(mask_path)
    covered = mask.sum() / mask.size * 100.0
    print(f"Wrote {out_path} ({covered:.2f}% directly covered by captures)")
    print(f"Wrote {mask_path}")

    if args.tag_materials:
        material_map = np.zeros((height, width), np.uint8)
        best = np.zeros((height, width), np.uint32)
        for material_id, votes in sorted(material_votes.items()):
            better = votes > best
            material_map[better] = material_id
            best[better] = votes[better]
        if args.materials_dilate:
            material_map = resistance_dilate_labels(
                material_map,
                material_map > 0,
                args.materials_dilate,
                out,
                resistance=args.dilate_resistance,
                target_mask=output_mask,
            )
        materials_path = Path(args.materials_output) if args.materials_output else out_path.with_name(out_path.stem + "_materials.png")
        labels_path = Path(args.materials_labels) if args.materials_labels else materials_path.with_suffix(".json")
        manifest_path = (
            Path(args.materials_manifest)
            if args.materials_manifest
            else out_path.with_name(out_path.stem + "_capture_materials_manifest.json")
        )
        for path in (materials_path, labels_path, manifest_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(material_map).save(materials_path)
        labels_path.write_text(json.dumps(material_labels_dict(), indent=1))
        manifest_path.write_text(
            json.dumps(
                {
                    "model": model,
                    "labels": material_labels_dict(),
                    "captures": capture_manifest,
                },
                indent=1,
            )
        )
        tagged = np.count_nonzero(material_map)
        print(f"Wrote {materials_path} ({tagged / material_map.size * 100.0:.2f}% tagged)")
        print(f"Wrote {labels_path}")
        print(f"Wrote {manifest_path}")
        print(
            "Semantic tags: "
            f"{counters['ai_calls']} AI calls, "
            f"{counters['dedup_hits']} dedup hits, "
            f"{counters['ai_fail']} fallbacks."
        )


def build_parser():
    ap = argparse.ArgumentParser(
        prog="pbr-surface",
        description="Generate PBR maps from baked mesh/Gaussian-splat diffuse UV atlases.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    atlas = sub.add_parser("atlas", help="generate normal/height/ORM from one atlas or a folder")
    atlas.add_argument("input", help="diffuse/base-color texture file or folder")
    atlas.add_argument("-o", "--output", default="./surface_pbr_out", help="output folder")
    atlas.add_argument("--mask", default="", help="optional mask image, or folder of masks")
    atlas.add_argument("--material-map", default="",
                       help="optional material-index PNG from bake-captures, or folder of maps")
    atlas.add_argument("--material-labels", default="",
                       help="optional JSON mapping material-map indices to material names")
    atlas.add_argument("--ignore-alpha", action="store_true", help="do not use input alpha as UV-valid mask")
    atlas.add_argument("--alpha-threshold", type=int, default=1, help="alpha/luma threshold for masks")
    atlas.add_argument("--write-mask", action="store_true", help="write the resolved valid-UV mask beside outputs")
    atlas.add_argument("--dilate-padding", type=int, default=16, help="fill invalid atlas padding to reduce mip bleed")
    atlas.add_argument("--seamless", action="store_true", help="use wrap-around filtering for tiled textures")
    atlas.add_argument("--shadow-suppression", type=float, default=0.25,
                       help="attenuate broad baked lighting before height inference (0..1)")
    atlas.add_argument("--api-url", default=os.environ.get("PBR_API_URL",
                       "http://localhost:1234/v1/chat/completions"),
                       help="OpenAI-compatible chat-completions URL")
    atlas.add_argument("--model", default=os.environ.get("PBR_MODEL", "auto"),
                       help="model id, or 'auto'")
    atlas.add_argument("--vision-max-tokens", type=int, default=1600,
                       help="max completion tokens for vision JSON classification")
    atlas.add_argument("--api-key", default=os.environ.get("PBR_API_KEY", ""),
                       help="bearer token for hosted endpoints")
    atlas.add_argument("--no-ai", action="store_true", help="skip the LLM; use color heuristics only")
    atlas.add_argument("--strength", type=float, default=1.25, help="global normal-relief multiplier")
    atlas.add_argument("--palette-colors", type=int, default=24, help="max perceptual clusters per atlas")
    atlas.add_argument("--dx", action="store_true", help="DirectX normals (flip green channel)")
    atlas.add_argument("--invert-height", action="store_true", help="treat bright pixels as low")
    atlas.add_argument("--no-filename-prior", action="store_true",
                       help="disable filename->material correction")
    atlas.add_argument("--reset-every", type=int, default=0,
                       help="reload model every N AI calls to flush image cache")
    atlas.add_argument("--reset-cmd", default='lms unload --all && lms load "{model}" -y',
                       help="shell command to reload the model; {model} is substituted")
    atlas.add_argument("--force", action="store_true", help="recompute even if outputs/cache exist")
    atlas.add_argument("--limit", type=int, default=0, help="process only first N files")
    atlas.set_defaults(func=atlas_command)

    bake = sub.add_parser("bake-captures", help="bake color+UV render-pass screenshots into one atlas")
    bake.add_argument("captures", help="folder containing capture pairs")
    bake.add_argument("-o", "--output", required=True, help="output baked diffuse atlas PNG")
    bake.add_argument("--mask-output", default="", help="output mask path (default: <output>_mask.png)")
    bake.add_argument("--size", default="2048", help="atlas size, e.g. 2048 or 4096x2048")
    bake.add_argument("--color-suffix", default="_color", help="color capture filename suffix")
    bake.add_argument("--uv-suffix", default="_uv", help="UV capture filename suffix")
    bake.add_argument("--flip-v", action="store_true", help="flip UV V while writing texture Y")
    bake.add_argument("--black-invalid", action="store_true",
                      help="treat black UV pixels as invalid when alpha is not available")
    bake.add_argument("--alpha-threshold", type=int, default=1, help="alpha threshold for valid UV pixels")
    bake.add_argument("--dilate", type=int, default=32, help="fill unsampled texels from neighbors")
    bake.add_argument("--dilate-mode", choices=("average", "resistance"), default="average",
                      help="UV fill mode: simple neighbor averaging or edge-aware resistance fill")
    bake.add_argument("--dilate-resistance", type=float, default=10.0,
                      help="edge resistance for --dilate-mode resistance and --materials-dilate")
    bake.add_argument("--expand-mask", action="store_true",
                      help="write the dilated/fill mask instead of only the directly sampled UV mask")
    bake.add_argument("--fill-target-mask", default="",
                      help="optional alpha/luma mask constraining UV fill to known atlas islands")
    bake.add_argument("--tag-materials", action="store_true",
                      help="classify each capture with the vision endpoint and bake material IDs into UV space")
    bake.add_argument("--segment-materials", action="store_true",
                      help="split each capture into region masks before material tagging")
    bake.add_argument("--segmenter", choices=("connected", "masks", "sam", "sam2"), default="connected",
                      help="region source for --segment-materials")
    bake.add_argument("--segment-masks", default="",
                      help="folder of external mask images for --segmenter masks")
    bake.add_argument("--segments-output", default="",
                      help="optional folder for per-capture segment/material debug PNGs")
    bake.add_argument("--segment-max-regions", type=int, default=32,
                      help="maximum regions tagged per capture; 0 means unlimited")
    bake.add_argument("--segment-min-area", type=int, default=256,
                      help="minimum segment area in screen pixels")
    bake.add_argument("--segment-min-coverage", type=float, default=0.002,
                      help="minimum segment coverage of valid capture pixels")
    bake.add_argument("--segment-palette-colors", type=int, default=48,
                      help="color clusters used by the built-in connected segmenter")
    bake.add_argument("--segment-region-colors", type=int, default=8,
                      help="palette swatches sent to the VLM for each region")
    bake.add_argument("--segment-crop-pad", type=int, default=16,
                      help="pixels of context around each masked region crop")
    bake.add_argument("--segment-fill-unassigned", action="store_true",
                      help="fallback-tag valid pixels not covered by any segment")
    bake.add_argument("--sam-checkpoint", default="", help="SAM checkpoint for --segmenter sam")
    bake.add_argument("--sam-model-type", default="vit_h",
                      help="SAM model type for --segmenter sam, e.g. vit_h/vit_l/vit_b")
    bake.add_argument("--sam2-checkpoint", default="", help="SAM2 checkpoint for --segmenter sam2")
    bake.add_argument("--sam2-config", default="", help="SAM2 model config for --segmenter sam2")
    bake.add_argument("--sam-device", default="", help="SAM/SAM2 device override, e.g. cuda or cpu")
    bake.add_argument("--sam-points-per-side", type=int, default=32,
                      help="SAM/SAM2 automatic mask generator sampling density")
    bake.add_argument("--sam-pred-iou-thresh", type=float, default=0.86,
                      help="SAM/SAM2 automatic mask predicted-IoU threshold")
    bake.add_argument("--sam-stability-score-thresh", type=float, default=0.92,
                      help="SAM/SAM2 automatic mask stability threshold")
    bake.add_argument("--materials-output", default="", help="material-index PNG output path")
    bake.add_argument("--materials-labels", default="", help="material-index labels JSON output path")
    bake.add_argument("--materials-manifest", default="", help="per-capture semantic manifest output path")
    bake.add_argument("--materials-dilate", type=int, default=0,
                      help="edge-aware fill of zero material IDs for N UV pixels, guided by the baked albedo")
    bake.add_argument("--palette-colors", type=int, default=24, help="max perceptual clusters per capture")
    bake.add_argument("--api-url", default=os.environ.get("PBR_API_URL",
                      "http://localhost:1234/v1/chat/completions"),
                      help="OpenAI-compatible chat-completions URL")
    bake.add_argument("--model", default=os.environ.get("PBR_MODEL", "auto"),
                      help="model id, or 'auto'")
    bake.add_argument("--vision-max-tokens", type=int, default=1600,
                      help="max completion tokens for vision JSON classification")
    bake.add_argument("--api-key", default=os.environ.get("PBR_API_KEY", ""),
                      help="bearer token for hosted endpoints")
    bake.add_argument("--no-ai", action="store_true", help="skip the LLM; use color heuristics only")
    bake.add_argument("--no-filename-prior", action="store_true",
                      help="disable filename->material correction")
    bake.add_argument("--reset-every", type=int, default=0,
                      help="reload model every N AI calls to flush image cache")
    bake.add_argument("--reset-cmd", default='lms unload --all && lms load "{model}" -y',
                      help="shell command to reload the model; {model} is substituted")
    bake.set_defaults(func=bake_captures_command)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
