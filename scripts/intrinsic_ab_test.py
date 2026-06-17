#!/usr/bin/env python3
"""
Compare intrinsic/de-lighting backends for the Gaussian-splat PBR pipeline.

The harness can run as a CLI smoke test or as a small Gradio app. It expects capture
pairs from pbr_surface.py: <name>_color.png and <name>_uv.png. Each backend writes a
de-lit albedo candidate, then the harness can feed that albedo into pbr_surface.py's
segmented material bake.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageChops, ImageStat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = ROOT / "artifacts" / "pbr-surface-validation" / "captures-front" / "google_towel_front_color.png"
DEFAULT_OUT = ROOT / "artifacts" / "intrinsic-ab-test"
PBR_SURFACE = Path(os.environ.get("PBR_SURFACE", r"F:\pbr-from-pixelart\pbr_surface.py"))
DEFAULT_API_URL = os.environ.get("PBR_API_URL", "http://127.0.0.1:1234/v1/chat/completions")
DEFAULT_VLM_MODEL = os.environ.get("PBR_MODEL", "qwen/qwen3-vl-30b")


@dataclass
class BackendResult:
    name: str
    ok: bool
    elapsed_s: float
    output_dir: Path
    files: dict[str, Path]
    note: str = ""


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def sibling_uv(color_path: Path, color_suffix: str = "_color", uv_suffix: str = "_uv") -> Path:
    stem = color_path.stem
    if not stem.endswith(color_suffix):
        raise ValueError(f"capture color filename must end with {color_suffix}: {color_path.name}")
    uv = color_path.with_name(stem[: -len(color_suffix)] + uv_suffix + color_path.suffix)
    if not uv.exists():
        raise FileNotFoundError(f"missing UV capture beside color image: {uv}")
    return uv


def image_delta_score(a: Image.Image, b: Image.Image) -> float:
    a = a.convert("RGB").resize(b.size, Image.Resampling.BILINEAR)
    diff = ImageChops.difference(a, b.convert("RGB"))
    stat = ImageStat.Stat(diff)
    return float(sum(stat.mean) / (3.0 * 255.0))


def write_metrics(result: BackendResult, source: Image.Image) -> None:
    metrics = {
        "backend": result.name,
        "ok": result.ok,
        "elapsed_s": result.elapsed_s,
        "note": result.note,
        "files": {k: str(v) for k, v in result.files.items()},
    }
    albedo_path = result.files.get("albedo")
    if albedo_path and albedo_path.exists():
        albedo = load_rgb(albedo_path)
        metrics["albedo_delta_vs_input"] = image_delta_score(source, albedo)
    (result.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def run_logged(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + subprocess.list2cmdline([str(c) for c in cmd]) + "\n\n")
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode:
        raise RuntimeError(f"command failed with exit code {proc.returncode}; see {log_path}")


def path_arg(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def merge_pythonpath(*paths: Path, env: dict[str, str] | None = None) -> dict[str, str]:
    merged = dict(os.environ if env is None else env)
    entries = [str(p) for p in paths if p.exists()]
    existing = merged.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    merged["PYTHONPATH"] = os.pathsep.join(entries)
    merged["PYTHONIOENCODING"] = "utf-8"
    return merged


def resized_copy(src: Path, dst: Path, size: tuple[int, int]) -> Path:
    img = load_rgb(src)
    if img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return save_image(img, dst)


def backend_identity(color_path: Path, output_dir: Path, **_: object) -> BackendResult:
    start = time.perf_counter()
    out = ensure_dir(output_dir / "identity")
    img = load_rgb(color_path)
    albedo = save_image(img, out / f"{color_path.stem}_albedo.png")
    gray = img.convert("L").convert("RGB")
    shading = save_image(gray, out / f"{color_path.stem}_shading.png")
    return BackendResult("identity", True, time.perf_counter() - start, out, {
        "albedo": albedo,
        "shading": shading,
    }, "Raw color pass baseline; no de-lighting.")


def run_marigold(
    color_path: Path,
    output_dir: Path,
    checkpoint: str,
    properties: tuple[str, ...],
    steps: int,
    processing_resolution: int,
    device: str,
) -> BackendResult:
    start = time.perf_counter()
    name = checkpoint.rsplit("/", 1)[-1]
    out = ensure_dir(output_dir / name)
    files: dict[str, Path] = {}
    try:
        import torch
        from diffusers import MarigoldIntrinsicsPipeline

        dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
        pipe = MarigoldIntrinsicsPipeline.from_pretrained(
            checkpoint,
            variant="fp16" if dtype == torch.float16 else None,
            torch_dtype=dtype,
        )
        pipe = pipe.to(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        image = load_rgb(color_path)
        prediction = pipe(
            image,
            num_inference_steps=steps,
            ensemble_size=1,
            processing_resolution=processing_resolution,
            match_input_resolution=True,
        )
        vis = pipe.image_processor.visualize_intrinsics(prediction.prediction, pipe.target_properties)[0]
        for prop in properties:
            if prop in vis:
                files[prop] = save_image(vis[prop].convert("RGB"), out / f"{color_path.stem}_{prop}.png")
        if "albedo" not in files:
            raise RuntimeError(f"Marigold output did not include albedo; got {sorted(vis.keys())}")
        return BackendResult(name, True, time.perf_counter() - start, out, files)
    except Exception as exc:
        note = f"{type(exc).__name__}: {exc}"
        (out / "error.txt").write_text(note)
        return BackendResult(name, False, time.perf_counter() - start, out, files, note)


def backend_marigold_lighting(color_path: Path, output_dir: Path, **kwargs: object) -> BackendResult:
    return run_marigold(
        color_path,
        output_dir,
        "prs-eth/marigold-iid-lighting-v1-1",
        ("albedo", "shading", "residual"),
        int(kwargs.get("steps", 4)),
        int(kwargs.get("processing_resolution", 768)),
        str(kwargs.get("device", "cuda")),
    )


def backend_marigold_appearance(color_path: Path, output_dir: Path, **kwargs: object) -> BackendResult:
    return run_marigold(
        color_path,
        output_dir,
        "prs-eth/marigold-iid-appearance-v1-1",
        ("albedo", "roughness", "metallicity"),
        int(kwargs.get("steps", 4)),
        int(kwargs.get("processing_resolution", 768)),
        str(kwargs.get("device", "cuda")),
    )


def external_backend_note(name: str, repo_dir: Path, output_dir: Path) -> BackendResult:
    start = time.perf_counter()
    out = ensure_dir(output_dir / name)
    if not repo_dir.exists():
        note = f"Repository not found: {repo_dir}"
    else:
        note = f"Repository found but no stable local adapter is wired yet: {repo_dir}"
    (out / "status.txt").write_text(note)
    return BackendResult(name, False, time.perf_counter() - start, out, {}, note)


def backend_intrinsicanything(color_path: Path, output_dir: Path, **_: object) -> BackendResult:
    start = time.perf_counter()
    repo = ROOT / "external" / "IntrinsicAnything"
    out = ensure_dir(output_dir / "intrinsicanything")
    files: dict[str, Path] = {}
    errors: list[str] = []
    try:
        python = repo / ".venv" / "Scripts" / "python.exe"
        if not python.exists():
            raise FileNotFoundError(f"IntrinsicAnything venv python not found: {python}")
        for checkpoint in ("albedo", "specular"):
            ckpt_dir = repo / "weights" / checkpoint
            if not (ckpt_dir / "checkpoints" / "last.ckpt").exists():
                raise FileNotFoundError(f"missing IntrinsicAnything checkpoint: {ckpt_dir}")

        input_dir = ensure_dir(out / "input")
        copied_input = input_dir / color_path.name
        shutil.copy2(color_path, copied_input)
        env = merge_pythonpath(
            repo / ".venv" / "src" / "taming-transformers",
            repo / ".venv" / "src" / "clip",
            repo / "models",
        )
        ddim = os.environ.get("INTRINSICANYTHING_DDIM", "25")
        for checkpoint, key in (("albedo", "albedo"), ("specular", "specular")):
            run_dir = ensure_dir(out / checkpoint)
            try:
                run_logged(
                    [
                        python,
                        "inference.py",
                        "--input_dir",
                        input_dir,
                        "--model_dir",
                        repo / "weights" / checkpoint,
                        "--output_dir",
                        run_dir,
                        "--ddim",
                        ddim,
                        "--batch_size",
                        "1",
                    ],
                    cwd=repo,
                    log_path=out / f"{checkpoint}.log",
                    env=env,
                )
                produced = run_dir / copied_input.name
                if not produced.exists():
                    raise FileNotFoundError(f"expected IntrinsicAnything output was not created: {produced}")
                files[key] = resized_copy(produced, out / f"{color_path.stem}_{key}.png", load_rgb(color_path).size)
            except Exception as exc:
                errors.append(f"{checkpoint}: {type(exc).__name__}: {exc}")
        ok = "albedo" in files
        note = "; ".join(errors)
        return BackendResult("intrinsicanything", ok, time.perf_counter() - start, out, files, note)
    except Exception as exc:
        note = f"{type(exc).__name__}: {exc}"
        (out / "error.txt").write_text(note)
        return BackendResult("intrinsicanything", False, time.perf_counter() - start, out, files, note)


def backend_iid(color_path: Path, output_dir: Path, **kwargs: object) -> BackendResult:
    start = time.perf_counter()
    repo = ROOT / "external" / "IntrinsicImageDiffusion"
    out = ensure_dir(output_dir / "intrinsic_image_diffusion")
    files: dict[str, Path] = {}
    try:
        python = repo / ".venv" / "Scripts" / "python.exe"
        if not python.exists():
            raise FileNotFoundError(f"IntrinsicImageDiffusion venv python not found: {python}")
        ckpt = repo / "models" / "material_diffusion" / "iid_e250.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing IntrinsicImageDiffusion material checkpoint: {ckpt}")
        env = merge_pythonpath(repo)
        run_logged(
            [
                python,
                "-m",
                "iid.material_diffusion",
                "logger=console",
                "+logger.plot_images=false",
                "+logger.save_images=false",
                f"data.input_path={path_arg(color_path)}",
                f"output.folder={path_arg(out)}",
                "output.as_dataset=false",
                "model.num_samples=1",
                "model.sampling_batch_size=1",
                f"device={str(kwargs.get('device', 'cuda'))}",
                f"hydra.run.dir={path_arg(out / 'hydra')}",
            ],
            cwd=repo,
            log_path=out / "intrinsic_image_diffusion.log",
            env=env,
        )
        target_size = load_rgb(color_path).size
        expected = {
            "albedo": out / f"{color_path.stem}_albedo.png",
            "roughness": out / f"{color_path.stem}_roughness.png",
            "metallicity": out / f"{color_path.stem}_metal.png",
        }
        for key, produced in expected.items():
            if not produced.exists():
                raise FileNotFoundError(f"expected IntrinsicImageDiffusion output was not created: {produced}")
            files[key] = resized_copy(produced, produced, target_size)
        return BackendResult("intrinsic_image_diffusion", True, time.perf_counter() - start, out, files)
    except Exception as exc:
        note = f"{type(exc).__name__}: {exc}"
        (out / "error.txt").write_text(note)
        return BackendResult("intrinsic_image_diffusion", False, time.perf_counter() - start, out, files, note)


BACKENDS: dict[str, Callable[..., BackendResult]] = {
    "identity": backend_identity,
    "marigold_lighting": backend_marigold_lighting,
    "marigold_appearance": backend_marigold_appearance,
    "intrinsicanything": backend_intrinsicanything,
    "intrinsic_image_diffusion": backend_iid,
}


def make_backend_capture(color_path: Path, uv_path: Path, result: BackendResult) -> Path | None:
    albedo = result.files.get("albedo")
    if not albedo or not albedo.exists():
        return None
    cap_dir = ensure_dir(result.output_dir / "captures")
    shutil.copy2(albedo, cap_dir / f"{color_path.stem}_albedo.png")
    shutil.copy2(uv_path, cap_dir / f"{color_path.stem}_uv.png")
    return cap_dir


def run_pbr_surface(
    color_path: Path,
    uv_path: Path,
    result: BackendResult,
    size: int,
    api_url: str,
    vlm_model: str,
    no_ai: bool,
) -> dict[str, Path]:
    cap_dir = make_backend_capture(color_path, uv_path, result)
    if cap_dir is None:
        return {}
    out_albedo = result.output_dir / "baked_albedo.png"
    cmd = [
        sys.executable,
        str(PBR_SURFACE),
        "bake-captures",
        str(cap_dir),
        "-o",
        str(out_albedo),
        "--size",
        str(size),
        "--color-suffix",
        "_albedo",
        "--uv-suffix",
        "_uv",
        "--dilate",
        "24",
        "--segment-materials",
        "--segmenter",
        "connected",
        "--segment-max-regions",
        "12",
        "--segment-fill-unassigned",
        "--segments-output",
        str(result.output_dir / "segments"),
        "--no-filename-prior",
    ]
    if no_ai:
        cmd.append("--no-ai")
    else:
        cmd.extend(["--api-url", api_url, "--model", vlm_model, "--vision-max-tokens", "1600"])
    subprocess.run(cmd, check=True)

    pbr_dir = ensure_dir(result.output_dir / "pbr")
    cmd = [
        sys.executable,
        str(PBR_SURFACE),
        "atlas",
        str(out_albedo),
        "-o",
        str(pbr_dir),
        "--mask",
        str(out_albedo.with_name(out_albedo.stem + "_mask.png")),
        "--material-map",
        str(out_albedo.with_name(out_albedo.stem + "_materials.png")),
        "--material-labels",
        str(out_albedo.with_name(out_albedo.stem + "_materials.json")),
        "--force",
    ]
    subprocess.run(cmd, check=True)
    return {
        "baked_albedo": out_albedo,
        "material_map": out_albedo.with_name(out_albedo.stem + "_materials.png"),
        "capture_manifest": out_albedo.with_name(out_albedo.stem + "_capture_materials_manifest.json"),
        "normal": pbr_dir / f"{out_albedo.stem}_n.png",
        "height": pbr_dir / f"{out_albedo.stem}_h.png",
        "orm": pbr_dir / f"{out_albedo.stem}_orm.png",
    }


def run_suite(
    color_path: Path,
    output_dir: Path,
    backends: list[str],
    run_pbr: bool,
    size: int,
    steps: int,
    processing_resolution: int,
    device: str,
    api_url: str,
    vlm_model: str,
    no_ai: bool,
) -> list[BackendResult]:
    color_path = color_path.resolve()
    uv_path = sibling_uv(color_path)
    output_dir = ensure_dir(output_dir)
    source = load_rgb(color_path)
    results = []
    for backend_name in backends:
        backend = BACKENDS[backend_name]
        result = backend(
            color_path,
            output_dir,
            steps=steps,
            processing_resolution=processing_resolution,
            device=device,
        )
        if run_pbr and result.ok:
            try:
                result.files.update(run_pbr_surface(color_path, uv_path, result, size, api_url, vlm_model, no_ai))
            except Exception as exc:
                result.note = (result.note + "\n" if result.note else "") + f"PBR failed: {type(exc).__name__}: {exc}"
        write_metrics(result, source)
        results.append(result)
    summary = [
        {
            "backend": r.name,
            "ok": r.ok,
            "elapsed_s": r.elapsed_s,
            "note": r.note,
            "output_dir": str(r.output_dir),
            "files": {k: str(v) for k, v in r.files.items()},
        }
        for r in results
    ]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return results


def launch_app(args: argparse.Namespace) -> None:
    import gradio as gr

    def run_from_ui(color_file, selected, run_pbr, steps, proc_res, size, no_ai):
        color_path = Path(color_file.name if hasattr(color_file, "name") else color_file)
        results = run_suite(
            color_path=color_path,
            output_dir=Path(args.output),
            backends=list(selected),
            run_pbr=bool(run_pbr),
            size=int(size),
            steps=int(steps),
            processing_resolution=int(proc_res),
            device=args.device,
            api_url=args.api_url,
            vlm_model=args.vlm_model,
            no_ai=bool(no_ai),
        )
        gallery = []
        table = []
        for r in results:
            for key in ("albedo", "shading", "residual", "roughness", "metallicity", "material_map", "orm", "normal"):
                p = r.files.get(key)
                if p and p.exists():
                    gallery.append((str(p), f"{r.name}: {key}"))
            table.append([r.name, r.ok, round(r.elapsed_s, 2), r.note, str(r.output_dir)])
        return gallery, table, str(Path(args.output) / "summary.json")

    with gr.Blocks(title="Intrinsic A/B Test") as demo:
        gr.Markdown("# Intrinsic / De-lighting A/B Test")
        color = gr.File(label="Capture color PNG", value=str(args.capture))
        selected = gr.CheckboxGroup(
            choices=list(BACKENDS.keys()),
            value=args.backends,
            label="Backends",
        )
        with gr.Row():
            run_pbr = gr.Checkbox(value=args.run_pbr, label="Run segmented PBR bake")
            no_ai = gr.Checkbox(value=args.no_ai, label="No VLM tagging")
        with gr.Row():
            steps = gr.Slider(1, 50, value=args.steps, step=1, label="Diffusion steps")
            proc_res = gr.Slider(256, 1536, value=args.processing_resolution, step=64, label="Processing resolution")
            size = gr.Slider(256, 4096, value=args.size, step=256, label="Bake atlas size")
        run = gr.Button("Run comparison")
        gallery = gr.Gallery(label="Outputs", columns=3, height=720)
        table = gr.Dataframe(headers=["backend", "ok", "seconds", "note", "output_dir"], label="Runs")
        summary = gr.Textbox(label="Summary JSON")
        run.click(run_from_ui, [color, selected, run_pbr, steps, proc_res, size, no_ai], [gallery, table, summary])
    demo.launch(server_name=args.host, server_port=args.port)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", default=str(DEFAULT_CAPTURE), help="input *_color.png capture")
    ap.add_argument("--output", default=str(DEFAULT_OUT), help="output directory")
    ap.add_argument("--backends", nargs="+", choices=list(BACKENDS.keys()),
                    default=list(BACKENDS.keys()))
    ap.add_argument("--run-pbr", action="store_true", help="run pbr_surface segmented bake for each albedo")
    ap.add_argument("--no-ai", action="store_true", help="use heuristic tagging in pbr_surface")
    ap.add_argument("--size", type=int, default=1024, help="baked atlas size for pbr_surface")
    ap.add_argument("--steps", type=int, default=4, help="Marigold denoising steps")
    ap.add_argument("--processing-resolution", type=int, default=768, help="Marigold processing resolution")
    ap.add_argument("--device", default="cuda", help="torch device")
    ap.add_argument("--api-url", default=DEFAULT_API_URL, help="OpenAI-compatible VLM endpoint")
    ap.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL, help="VLM model for material tagging")
    ap.add_argument("--app", action="store_true", help="launch Gradio app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7862)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.app:
        launch_app(args)
    else:
        results = run_suite(
            color_path=Path(args.capture),
            output_dir=Path(args.output),
            backends=args.backends,
            run_pbr=args.run_pbr,
            size=args.size,
            steps=args.steps,
            processing_resolution=args.processing_resolution,
            device=args.device,
            api_url=args.api_url,
            vlm_model=args.vlm_model,
            no_ai=args.no_ai,
        )
        for result in results:
            status = "ok" if result.ok else "failed"
            print(f"{result.name}: {status} in {result.elapsed_s:.1f}s -> {result.output_dir}")
            if result.note:
                print(f"  {result.note}")


if __name__ == "__main__":
    main()
