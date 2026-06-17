#!/usr/bin/env python3
# Copyright (C) 2026 Gregor Hubert Max Koch
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
pbr_pixelart -- generate full PBR material maps from pixel-art diffuse textures.

For every diffuse tile <name>.png it writes:
    <name>_n.png    normal map      (structure-aware, seamless)
    <name>_h.png    height map      (structure-aware, seamless)
    <name>_orm.png  packed ORM      R=AmbientOcclusion  G=Roughness  B=Metallic

DESIGN
------
A vision model can't reliably segment a 32x32 image pixel-by-pixel, and it can't read
roughness/metallic from colour. So the work is split:

  * GROUND TRUTH (deterministic): the tile's quantized palette. Per-pixel material =
    palette-index lookup. This drives roughness / metallic / per-material relief exactly.

  * STRUCTURE (the model's job): a vision LLM returns ONE JSON object naming the material
    of every palette colour PLUS structural hints (pattern, grain, relief, recessed/raised
    edges, notable features). These fold into HEIGHT generation -- mortar lines become real
    grooves, rivets become bumps -- instead of guessing everything from luminance.

  Normal + AO are derived from the structure-aware height with wrap-around convolution
  (seamless tiling).

BACKEND
-------
Works with ANY OpenAI-compatible chat-completions endpoint that accepts image input:
LM Studio, Ollama, llama.cpp server, vLLM, OpenRouter, OpenAI, etc.
Configure with --api-url / --model / --api-key. --model auto picks the first served model.

OPTIONAL diffusion normals (--normals stablenormal) require the extra deps in
requirements-diffusion.txt. NOTE: for tiny pixel art the default procedural normals are
crisper; diffusion only helps at 512px+.

Deps: pip install pillow numpy requests tqdm
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Material table -- bridge from "what the AI names" to PBR numbers.
#   rough/metal : the ORM values.   nstr : per-material relief.   ao : AO strength.
# Tune freely; this is where the look lives.
# --------------------------------------------------------------------------------------
MATERIALS = {
    "wood":         dict(rough=0.80, metal=0.0, nstr=1.0, ao=1.0),
    "stone":        dict(rough=0.90, metal=0.0, nstr=1.6, ao=1.3),
    "brick":        dict(rough=0.88, metal=0.0, nstr=1.5, ao=1.4),
    "dirt":         dict(rough=0.95, metal=0.0, nstr=1.2, ao=1.2),
    "sand":         dict(rough=0.92, metal=0.0, nstr=0.8, ao=0.9),
    "gravel":       dict(rough=0.95, metal=0.0, nstr=1.6, ao=1.4),
    "grass":        dict(rough=0.90, metal=0.0, nstr=1.1, ao=1.1),
    "plant":        dict(rough=0.85, metal=0.0, nstr=1.0, ao=1.0),
    "leaf":         dict(rough=0.80, metal=0.0, nstr=0.9, ao=1.0),
    "wool":         dict(rough=0.97, metal=0.0, nstr=0.8, ao=1.0),
    "metal_iron":   dict(rough=0.40, metal=1.0, nstr=0.7, ao=0.6),
    "metal_steel":  dict(rough=0.30, metal=1.0, nstr=0.6, ao=0.5),
    "metal_gold":   dict(rough=0.25, metal=1.0, nstr=0.5, ao=0.5),
    "metal_copper": dict(rough=0.30, metal=1.0, nstr=0.6, ao=0.5),
    "metal_silver": dict(rough=0.22, metal=1.0, nstr=0.5, ao=0.5),
    "rust":         dict(rough=0.95, metal=0.0, nstr=1.3, ao=1.2),  # oxidized = dielectric
    "cloth":        dict(rough=0.95, metal=0.0, nstr=0.9, ao=1.0),
    "leather":      dict(rough=0.70, metal=0.0, nstr=1.0, ao=1.0),
    "water":        dict(rough=0.10, metal=0.0, nstr=0.3, ao=0.3),
    "ice":          dict(rough=0.15, metal=0.0, nstr=0.4, ao=0.4),
    "snow":         dict(rough=0.60, metal=0.0, nstr=0.6, ao=0.7),
    "crystal":      dict(rough=0.20, metal=0.0, nstr=1.4, ao=0.7),
    "gem":          dict(rough=0.15, metal=0.0, nstr=1.5, ao=0.7),
    "glass":        dict(rough=0.10, metal=0.0, nstr=0.5, ao=0.3),
    "plastic":      dict(rough=0.45, metal=0.0, nstr=0.6, ao=0.7),
    "ceramic":      dict(rough=0.35, metal=0.0, nstr=0.7, ao=0.7),
    "bone":         dict(rough=0.70, metal=0.0, nstr=1.0, ao=1.0),
    "skin":         dict(rough=0.60, metal=0.0, nstr=0.8, ao=0.9),
    "lava":         dict(rough=0.85, metal=0.0, nstr=1.2, ao=1.1),
    "paper":        dict(rough=0.85, metal=0.0, nstr=0.6, ao=0.8),
    "unknown":      dict(rough=0.70, metal=0.0, nstr=1.0, ao=1.0),
}
MATERIAL_NAMES = list(MATERIALS.keys())

# Filename -> base material prior. Standardized texture-pack names are very reliable.
# Tiles whose name contains "ore" are skipped (the AI keeps embedded specks). Extend at will.
FILENAME_HINTS = {
    "leaves": "leaf", "leaf": "leaf", "sapling": "leaf", "vine": "leaf",
    "cactus": "plant", "reeds": "plant", "mushroom": "plant", "flower": "plant",
    "grass": "grass", "tallgrass": "grass", "fern": "grass",
    "planks": "wood", "plank": "wood", "log": "wood", "wood": "wood", "tree": "wood",
    "bookshelf": "wood", "fence": "wood", "crafting": "wood",
    "sandstone": "stone", "stonebrick": "stone", "cobblestone": "stone", "cobble": "stone",
    "smoothstone": "stone", "whitestone": "stone", "andesite": "stone", "granite": "stone",
    "diorite": "stone", "obsidian": "stone", "bedrock": "stone", "netherrack": "stone",
    "hellrock": "stone", "endstone": "stone", "quartz": "stone", "furnace": "stone",
    "stoneslab": "stone", "stonemoss": "stone", "redstone": "stone", "stone": "stone",
    "brick": "brick",
    "soulsand": "sand", "redsand": "sand", "hellsand": "sand", "sand": "sand",
    "gravel": "gravel",
    "podzol": "dirt", "mycelium": "dirt", "mycel": "dirt", "farmland": "dirt", "dirt": "dirt",
    "hardenedclay": "ceramic", "terracotta": "ceramic", "clay": "ceramic",
    "snow": "snow", "packedice": "ice", "ice": "ice", "glass": "glass",
    "wool": "wool", "carpet": "wool", "cloth": "wool", "web": "cloth",
    "water": "water", "lava": "lava",
    "goldblock": "metal_gold", "gold": "metal_gold",
    "ironblock": "metal_iron", "iron": "metal_iron",
    "diamondblock": "gem", "diamond": "gem", "emerald": "gem", "lapis": "gem",
    "glowstone": "crystal", "lightgem": "crystal", "sealantern": "crystal",
    "bone": "bone", "skull": "bone", "paper": "paper", "book": "paper",
}
# AI labels we treat as "confused base" and let the filename override (accents are kept):
CONFUSABLE = {"stone", "dirt", "snow", "unknown", "ceramic"}

RELIEF_SCALE = {"flat": 0.4, "low": 0.7, "medium": 1.0, "high": 1.5}
RAISED_FEATURES = {"rivet", "stud", "bump", "gem", "boss", "highlight", "nub"}
SUNKEN_FEATURES = {"crack", "hole", "dent", "knot", "scratch", "groove", "pit"}

SWATCH = 48

SYSTEM_PROMPT = (
    "You analyze a single pixel-art TILE for a PBR texture pipeline. The image shows the "
    "tile (upscaled) on top and its numbered color palette below (each swatch is labeled "
    "with its index and % coverage). The tile pixel coordinates are col,row (col=x "
    "left->right, row=y top->bottom).\n"
    "IMPORTANT: different palette colors often represent DIFFERENT materials in the same "
    "tile -- e.g. metallic ore specks embedded in gray stone, moss on stone, or dirt under "
    "grass. Judge EACH color individually by its hue and coverage; do NOT assign every "
    "swatch the same material unless the tile genuinely is a single material.\n"
    "Return ONE JSON object, no prose, with these keys:\n"
    '  "palette": object mapping each swatch number (string) to a material from the '
    "ALLOWED list -- the most important field; a small bright/odd-hued swatch among a "
    "dominant one is usually an embedded accent (ore, gem, moss), not the base.\n"
    '  "material": the dominant material (from ALLOWED).\n'
    '  "pattern": one of [planks, bricks, tiles, cobble, scales, grains, crystals, '
    "noise, smooth, woven, organic].\n"
    '  "grain": dominant line/seam direction, one of [horizontal, vertical, diagonal, none].\n'
    '  "relief": overall surface bumpiness, one of [flat, low, medium, high].\n'
    '  "edges": are boundaries between different colors recessed grooves, raised ridges, '
    "or neither -- one of [recessed, raised, none].\n"
    '  "features": array (<=6) of notable points, each {"type":..., "col":int, "row":int}; '
    "type in [rivet, stud, bump, gem, knot, crack, hole, scratch, highlight]; [] if none.\n"
    "ALLOWED materials: " + ", ".join(MATERIAL_NAMES) + "\n"
    'Example: {"palette":{"0":"stone","1":"metal_iron"},"material":"stone",'
    '"pattern":"cobble","grain":"none","relief":"high","edges":"recessed","features":[]}'
)


# ======================================================================================
# Palette
# ======================================================================================
def extract_palette(img_rgb, ncolors):
    """Quantize to <=ncolors perceptual clusters; return (idx_map, palette, coverage).
    Indices sorted by coverage (0 = most common). dither=NONE keeps clusters hard."""
    n_unique = len(img_rgb.getcolors(maxcolors=1 << 24) or [])
    colors = min(ncolors, max(2, n_unique))
    p = img_rgb.convert("P", palette=Image.ADAPTIVE, colors=colors, dither=Image.Dither.NONE)
    idx = np.array(p, dtype=np.int32)
    pal = p.getpalette()[: 256 * 3]
    counts = np.bincount(idx.reshape(-1), minlength=256)
    used = [u for u in np.argsort(-counts) if counts[u] > 0]
    remap = {old: new for new, old in enumerate(used)}
    idx = np.vectorize(remap.get)(idx).astype(np.int32)
    palette = [(pal[u * 3], pal[u * 3 + 1], pal[u * 3 + 2]) for u in used]
    total = idx.size
    coverage = [counts[u] / total for u in used]
    return idx, palette, coverage


# ======================================================================================
# Vision
# ======================================================================================
def build_prompt_image(img_rgb, palette, coverage, upscale=256):
    tile = img_rgb.resize((upscale, upscale), Image.NEAREST)
    cols = min(len(palette), 6)
    rows = (len(palette) + cols - 1) // cols
    canvas = Image.new("RGB", (max(upscale, cols * SWATCH), upscale + rows * SWATCH + 8),
                       (30, 30, 30))
    canvas.paste(tile, ((canvas.width - upscale) // 2, 0))
    d = ImageDraw.Draw(canvas)
    for i, c in enumerate(palette):
        cx, cy = (i % cols) * SWATCH, upscale + 8 + (i // cols) * SWATCH
        d.rectangle([cx, cy, cx + SWATCH - 2, cy + SWATCH - 2], fill=c, outline=(255, 255, 255))
        lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        fg = (0, 0, 0) if lum > 110 else (255, 255, 255)
        d.text((cx + 3, cy + 2), f"{i}", fill=fg)
        d.text((cx + 3, cy + 14), f"{coverage[i]*100:.0f}%", fill=fg)
    return canvas


def palette_text(palette, coverage):
    return "; ".join(f"#{i} rgb({r},{g},{b}) {coverage[i]*100:.0f}%"
                     for i, (r, g, b) in enumerate(palette))


def encode_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def clean_name_hint(stem):
    h = re.sub(r"[\(\)\d]+", " ", stem)           # drop "(7)" style dup markers
    h = re.sub(r"[_\-]+", " ", h).strip()
    return h


def extract_json_object(text):
    """Extract one JSON object from chat output, tolerating common local-model wrappers."""
    txt = (text or "").replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    txt = re.sub(r"```(?:json)?|```", "", txt, flags=re.I)
    start = txt.find("{")
    if start < 0:
        raise ValueError(f"no JSON: {txt[:160]}")

    depth = 0
    in_str = False
    esc = False
    end = -1
    for pos in range(start, len(txt)):
        ch = txt[pos]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = pos + 1
                    break
    if end < 0:
        last = txt.rfind("}")
        if last <= start:
            raise ValueError(f"incomplete JSON: {txt[start:start + 160]}")
        raw = txt[start:last + 1]
    else:
        raw = txt[start:end]
    repairs = [
        raw,
        re.sub(r'("features"\s*:\s*\[\])"', r"\1", raw),
        re.sub(r",\s*([}\]])", r"\1", raw),
    ]
    repairs.append(re.sub(r",\s*([}\]])", r"\1", repairs[1]))
    last_err = None
    for candidate in repairs:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_err = e
    raise ValueError(f"bad JSON: {last_err}: {raw[:160]}")


def filename_material(stem):
    """Longest-keyword base-material match, or None. Defers to AI on 'ore' names."""
    key = re.sub(r"[^a-z]", "", stem.lower())
    if "ore" in key:
        return None
    best = None
    for kw, mat in FILENAME_HINTS.items():
        if kw in key and (best is None or len(kw) > len(best[0])):
            best = (kw, mat)
    return best[1] if best else None


def apply_filename_prior(res, stem):
    """Override confused/base palette labels with the filename's material; keep accents."""
    base = filename_material(stem)
    if not base:
        return res
    dom_ai = res["material"]
    for i in list(res["palette"]):
        m = res["palette"][i]
        if m == dom_ai or m in CONFUSABLE:
            res["palette"][i] = base
    res["material"] = base
    return res


def classify(img_rgb, palette, coverage, name_hint, api_url, model, headers, timeout=120):
    prompt_img = build_prompt_image(img_rgb, palette, coverage)
    user_txt = (f"Filename hint (usually reliable): '{name_hint}'.\n"
                f"Palette ({len(palette)} colors, index rgb coverage): "
                f"{palette_text(palette, coverage)}")
    payload = {
        "model": model, "temperature": 0, "max_tokens": 600,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_txt},
                {"type": "image_url", "image_url": {"url": encode_data_url(prompt_img)}},
            ]},
        ],
    }
    r = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    try:
        raw = extract_json_object(msg.get("content", ""))
    except ValueError:
        raw = extract_json_object(msg.get("reasoning_content", ""))
    return normalize_result(raw, palette)


def normalize_result(raw, palette):
    pal_in = raw.get("palette", {}) or {}
    palette_map = {}
    for i in range(len(palette)):
        name = str(pal_in.get(str(i), "unknown")).strip().lower()
        palette_map[i] = name if name in MATERIALS else "unknown"
    feats = []
    for f in (raw.get("features") or [])[:6]:
        try:
            feats.append({"type": str(f["type"]).lower(),
                          "col": int(f["col"]), "row": int(f["row"])})
        except (KeyError, ValueError, TypeError):
            pass
    return {
        "palette": palette_map,
        "material": str(raw.get("material", "unknown")).lower(),
        "pattern": str(raw.get("pattern", "noise")).lower(),
        "grain": str(raw.get("grain", "none")).lower(),
        "relief": raw.get("relief", "medium") if raw.get("relief") in RELIEF_SCALE else "medium",
        "edges": raw.get("edges", "none") if raw.get("edges") in ("recessed", "raised", "none") else "none",
        "features": feats,
    }


def heuristic_result(palette):
    """No-AI fallback: rough guess from colour alone."""
    pm = {}
    for i, (r, g, b) in enumerate(palette):
        mx, mn = max(r, g, b), min(r, g, b)
        sat = (mx - mn) / (mx + 1e-6)
        if b > r and b > g and sat > 0.2:   pm[i] = "water"
        elif g > r and g > b and sat > 0.2: pm[i] = "grass"
        elif r > 120 and g > 90 and b < 90 and sat > 0.25: pm[i] = "wood"
        elif sat < 0.12:                    pm[i] = "stone"
        else:                               pm[i] = "unknown"
    return {"palette": pm, "material": "unknown", "pattern": "noise",
            "grain": "none", "relief": "medium", "edges": "recessed", "features": []}


# ======================================================================================
# Map generation (numpy, wrap-around = seamless)
# ======================================================================================
def _roll_blur(a):
    acc = np.zeros_like(a)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += np.roll(np.roll(a, dy, 0), dx, 1)
    return acc / 9.0


def boundary_strength(idx_map):
    """0..1 per pixel: how many wrap-around neighbours differ in palette index."""
    b = np.zeros(idx_map.shape, np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        b += (idx_map != np.roll(np.roll(idx_map, dy, 0), dx, 1)).astype(np.float32)
    return b / 4.0


def add_dab(height, row, col, amp, sigma=1.1):
    H, W = height.shape
    rad = 2
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            w = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
            height[(row + dy) % H, (col + dx) % W] += amp * w


def build_pbr(img_rgb, idx_map, res, gstrength, directx, invert_height):
    rgb = np.array(img_rgb, dtype=np.float32)
    H, W = idx_map.shape
    assignment = res["palette"]
    relief = RELIEF_SCALE.get(res["relief"], 1.0)

    rough = np.empty((H, W), np.float32); metal = np.empty((H, W), np.float32)
    nstr = np.empty((H, W), np.float32);  aost = np.empty((H, W), np.float32)
    for i, mat in assignment.items():
        m = MATERIALS.get(mat, MATERIALS["unknown"]); mask = idx_map == i
        rough[mask], metal[mask] = m["rough"], m["metal"]
        nstr[mask], aost[mask] = m["nstr"], m["ao"]

    # --- structure-aware HEIGHT ---
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
    height = (1.0 - lum) if invert_height else lum
    height = 0.5 + (height - 0.5) * 0.6                    # tame raw luminance contrast

    if res["edges"] != "none":
        bs = _roll_blur(boundary_strength(idx_map))
        depth = 0.35 * relief * (1 if res["edges"] == "raised" else -1)
        height = np.clip(height + bs * depth, 0, 1)

    sx, sy = W / 32.0, H / 32.0
    for f in res["features"]:
        amp = 0.30 * relief
        if f["type"] in SUNKEN_FEATURES:   amp = -amp
        elif f["type"] not in RAISED_FEATURES: amp *= 0.4
        add_dab(height, int(round(f["row"] * sy)), int(round(f["col"] * sx)), amp)
    height = np.clip(height, 0, 1)

    # --- NORMAL from height (grain-anisotropic, per-material relief) ---
    gx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * 0.5
    gy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * 0.5
    if res["grain"] == "horizontal":   gx *= 0.8; gy *= 1.3
    elif res["grain"] == "vertical":   gx *= 1.3; gy *= 0.8
    strength = nstr * gstrength * relief
    nx, ny, nz = -gx * strength, (gy if directx else -gy) * strength, np.ones_like(height)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = (np.stack([nx * inv, ny * inv, nz * inv], -1) * 0.5 + 0.5)
    normal = (normal * 255).clip(0, 255).astype(np.uint8)

    # --- AO from cavity ---
    blurred = _roll_blur(_roll_blur(height))
    ao = np.clip(1.0 - np.clip((blurred - height) * 2.0 * aost, 0, 1), 0, 1)

    height_png = (height * 255).clip(0, 255).astype(np.uint8)
    orm = (np.stack([ao, rough, metal], -1) * 255).clip(0, 255).astype(np.uint8)  # R=AO G=R B=M
    return normal, height_png, orm


# ======================================================================================
# Endpoint helpers (model discovery + optional context reset)
# ======================================================================================
def api_base(api_url):
    return api_url.split("/v1/")[0].split("/api/")[0].rstrip("/")


def list_models(api_url, headers):
    r = requests.get(api_base(api_url) + "/v1/models", headers=headers, timeout=10)
    r.raise_for_status()
    return [m.get("id") for m in r.json().get("data", [])]


def resolve_model(spec, api_url, headers):
    if spec and spec.lower() != "auto":
        return spec
    try:
        ids = [i for i in list_models(api_url, headers) if i]
    except Exception as e:
        sys.exit(f"--model auto failed to query {api_url}: {e}\nPass --model explicitly.")
    if not ids:
        sys.exit("--model auto found no models on the endpoint; pass --model explicitly.")
    for kw in ("vl", "vision", "llava", "pixtral", "minicpm", "qwen", "gemma", "intern"):
        for i in ids:
            if kw in i.lower():
                return i
    return ids[0]


def model_ready(api_url, model, headers):
    """LM Studio exposes loaded-state at /api/v0/models; otherwise fall back to /v1/models."""
    base = api_base(api_url)
    try:
        r = requests.get(base + "/api/v0/models", headers=headers, timeout=6)
        if r.ok and r.json().get("data"):
            return any(m.get("id") == model and m.get("state") == "loaded"
                       for m in r.json()["data"])
    except Exception:
        pass
    try:
        return model in list_models(api_url, headers)
    except Exception:
        return False


def reset_context(api_url, model, headers, reset_cmd, wait=90):
    """Flush a server's vision/image cache by reloading the model (LM Studio: ~30MB ceiling)."""
    tqdm.write("  [reset] flushing model/vision cache...")
    subprocess.run(reset_cmd.format(model=model), shell=True, capture_output=True)
    for _ in range(wait):
        if model_ready(api_url, model, headers):
            return True
        time.sleep(1)
    tqdm.write("  [reset] WARNING: model not confirmed ready; continuing")
    return False


# ======================================================================================
# Persistence
# ======================================================================================
def file_hash(p):
    return hashlib.md5(p.read_bytes()).hexdigest()[:16]


def copy_res(res):
    return {"palette": dict(res["palette"]), "material": res["material"],
            "pattern": res["pattern"], "grain": res["grain"], "relief": res["relief"],
            "edges": res["edges"], "features": [dict(f) for f in res["features"]]}


def load_manifest(man_path):
    """Load manifest; quarantine a corrupt file (e.g. truncated by a crash) and start fresh."""
    if not man_path.exists():
        return {}
    try:
        return json.loads(man_path.read_text())
    except (json.JSONDecodeError, ValueError):
        bad = man_path.with_suffix(".json.corrupt")
        try:
            os.replace(man_path, bad)
            print(f"WARNING: corrupt manifest quarantined -> {bad}; starting fresh "
                  "(existing output PNGs are still used for resume).")
        except OSError:
            pass
        return {}


def _store(manifest, man_path, key, h, res):
    store = copy_res(res); store["palette"] = {str(k): v for k, v in res["palette"].items()}
    manifest[key] = {"hash": h, "result": store}
    tmp = man_path.with_suffix(".json.tmp")                   # atomic write: crash-safe
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, man_path)


# ======================================================================================
# CLI
# ======================================================================================
def build_parser():
    ap = argparse.ArgumentParser(
        prog="pbr-pixelart",
        description="Generate PBR maps (normal / height / ORM) from pixel-art diffuse textures.")
    ap.add_argument("input_dir", help="folder of diffuse textures (scanned recursively)")
    ap.add_argument("-o", "--output", default="./pbr_out", help="output folder (default ./pbr_out)")
    # endpoint
    ap.add_argument("--api-url", default=os.environ.get("PBR_API_URL",
                    "http://localhost:1234/v1/chat/completions"),
                    help="OpenAI-compatible chat-completions URL (env PBR_API_URL)")
    ap.add_argument("--model", default=os.environ.get("PBR_MODEL", "auto"),
                    help="model id, or 'auto' to pick the first served vision model (env PBR_MODEL)")
    ap.add_argument("--api-key", default=os.environ.get("PBR_API_KEY", ""),
                    help="bearer token for hosted endpoints (env PBR_API_KEY)")
    ap.add_argument("--no-ai", action="store_true", help="skip the LLM; use colour heuristics only")
    # look
    ap.add_argument("--strength", type=float, default=1.5, help="global normal-relief multiplier")
    ap.add_argument("--palette-colors", type=int, default=10, help="max perceptual clusters per tile")
    ap.add_argument("--dx", action="store_true", help="DirectX normals (flip green channel)")
    ap.add_argument("--invert-height", action="store_true", help="treat bright pixels as low, not high")
    ap.add_argument("--no-filename-prior", action="store_true",
                    help="disable filename->material correction")
    # normals backend
    ap.add_argument("--normals", choices=["procedural", "stablenormal"], default="procedural",
                    help="normal backend (procedural is best for tiny pixel art)")
    ap.add_argument("--sn-cache", default="./sn_cache", help="StableNormal weights/cache dir")
    ap.add_argument("--sn-res", type=int, default=512, help="StableNormal working resolution")
    # cache flush (LM Studio etc.)
    ap.add_argument("--reset-every", type=int, default=0,
                    help="reload model every N AI calls to flush image cache (0=off; LM Studio: try 50)")
    ap.add_argument("--reset-cmd", default='lms unload --all && lms load "{model}" -y',
                    help="shell command to reload the model; {model} is substituted")
    # run control
    ap.add_argument("--force", action="store_true", help="recompute even if outputs/cache exist")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N files (testing)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    in_dir, out_dir = Path(args.input_dir), Path(args.output)
    if not in_dir.is_dir():
        sys.exit(f"input_dir not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "_materials_manifest.json"
    manifest = load_manifest(man_path)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    exts = {".png", ".bmp", ".gif", ".tga", ".jpg", ".jpeg"}
    files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in exts)
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"No images in {in_dir}")

    model = "(heuristic)" if args.no_ai else resolve_model(args.model, args.api_url, headers)
    print(f"{len(files)} textures -> {out_dir}  (model: {model}, normals: {args.normals})")

    sn = None
    if args.normals == "stablenormal":
        import sn_backend as sn
        print("Loading StableNormal (first run downloads weights)...")
        sn.load_predictor(args.sn_cache)

    ai_calls = ai_fail = dedup_hits = skipped = 0
    content_cache = {}          # pixel-hash -> raw (pre-prior) classification, this run
    for p in tqdm(files, unit="tex"):
        try:
            stem = out_dir / p.stem
            outs = (f"{stem}_n.png", f"{stem}_h.png", f"{stem}_orm.png")
            if not args.force and all(os.path.exists(o) for o in outs):
                skipped += 1                       # maps already on disk (crash-safe resume)
                continue

            img = Image.open(p).convert("RGB")
            idx_map, palette, coverage = extract_palette(img, args.palette_colors)
            h = file_hash(p)
            chash = hashlib.md5(img.tobytes()).hexdigest()[:16]
            key = str(p.relative_to(in_dir)).replace("\\", "/")

            if not args.force and manifest.get(key, {}).get("hash") == h:
                res = manifest[key]["result"]                       # cached / hand-edited
                res["palette"] = {int(k): v for k, v in res["palette"].items()}
            else:
                if args.no_ai:
                    raw = heuristic_result(palette)
                elif chash in content_cache:
                    raw = content_cache[chash]; dedup_hits += 1
                else:
                    try:
                        raw = classify(img, palette, coverage, clean_name_hint(p.stem),
                                       args.api_url, model, headers)
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
                _store(manifest, man_path, key, h, res)

            normal, height, orm = build_pbr(img, idx_map, res, args.strength,
                                            args.dx, args.invert_height)
            if sn is not None:
                try:
                    normal = sn.diffusion_normal(img, target_res=args.sn_res, directx=args.dx)
                except Exception as e:
                    tqdm.write(f"  StableNormal fail {p.name}: {e} -> procedural normal")
            Image.fromarray(normal).save(f"{stem}_n.png")
            Image.fromarray(height).save(f"{stem}_h.png")
            Image.fromarray(orm).save(f"{stem}_orm.png")
        except Exception as e:
            tqdm.write(f"  ERROR {p.name}: {e}")

    print(f"Done. {len(files)} files: {skipped} already done (skipped), "
          f"{ai_calls} AI calls, {dedup_hits} dedup hits, {ai_fail} fallbacks.")
    print(f"Manifest (editable): {man_path}")


if __name__ == "__main__":
    main()
