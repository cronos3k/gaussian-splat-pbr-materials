#!/usr/bin/env python3
# Copyright (C) 2026 Gregor Hubert Max Koch
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_obj(obj_path):
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(obj_path))
    else:
        bpy.ops.import_scene.obj(filepath=str(obj_path))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh imported from {obj_path}")
    return meshes


def bounds_for(objects):
    corners = []
    for obj in objects:
        corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    low = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    high = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return low, high, (low + high) * 0.5, high - low


def assign_color_material(objects, texture_path):
    mat = bpy.data.materials.new("capture_albedo")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(texture_path))
    emission = nodes.new("ShaderNodeEmission")
    out = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
    mat.node_tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def assign_uv_material(objects):
    mat = bpy.data.materials.new("capture_uv_rg")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    texcoord = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    combine = nodes.new("ShaderNodeCombineColor")
    emission = nodes.new("ShaderNodeEmission")
    out = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(texcoord.outputs["UV"], separate.inputs["Vector"])
    mat.node_tree.links.new(separate.outputs["X"], combine.inputs["Red"])
    mat.node_tree.links.new(separate.outputs["Y"], combine.inputs["Green"])
    mat.node_tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    mat.node_tree.links.new(combine.outputs["Color"], emission.inputs["Color"])
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(mat)


VIEW_DIRECTIONS = {
    "front": Vector((0.25, -1.0, 0.35)),
    "back": Vector((-0.25, 1.0, 0.35)),
    "left": Vector((-1.0, -0.15, 0.25)),
    "right": Vector((1.0, 0.15, 0.25)),
    "top": Vector((0.0, -0.25, 1.0)),
    "bottom": Vector((0.0, 0.25, -1.0)),
}


def fibonacci_sphere_directions(count):
    """Uniform-ish full-sphere camera directions for overlapping multiview capture."""
    if count < 1:
        raise ValueError("view count must be >= 1")
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    directions = []
    for i in range(count):
        z = 1.0 - (2.0 * (i + 0.5) / count)
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        directions.append(Vector((math.cos(theta) * radius, math.sin(theta) * radius, z)))
    return directions


def parse_views(spec):
    views = []
    for token in [part.strip().lower() for part in spec.split(",") if part.strip()]:
        if token in VIEW_DIRECTIONS:
            views.append((token, VIEW_DIRECTIONS[token]))
            continue
        for prefix in ("sphere", "orbit", "fibonacci"):
            if token.startswith(prefix):
                suffix = token[len(prefix):]
                count = int(suffix) if suffix else 32
                for i, direction in enumerate(fibonacci_sphere_directions(count)):
                    views.append((f"{prefix}{count}_{i:02d}", direction))
                break
        else:
            known = ", ".join(sorted(VIEW_DIRECTIONS))
            raise SystemExit(
                f"unknown view '{token}'. Known: {known}; generated: sphere32, orbit32, fibonacci32"
            )
    return views


def setup_camera(objects, ortho_scale_mult):
    low, high, center, size = bounds_for(objects)
    largest = max(size.x, size.y, size.z)
    cam_data = bpy.data.cameras.new("capture_camera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = largest * ortho_scale_mult
    cam = bpy.data.objects.new("capture_camera", cam_data)
    bpy.context.collection.objects.link(cam)

    # Front-ish view with enough elevation to reveal folds and avoid a dead-flat silhouette.
    direction = Vector((0.25, -1.0, 0.35)).normalized()
    cam.location = center + direction * largest * 3.0
    cam.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam

    light_data = bpy.data.lights.new("soft_area", type="AREA")
    light_data.energy = 350
    light_data.size = largest * 2.0
    light = bpy.data.objects.new("soft_area", light_data)
    bpy.context.collection.objects.link(light)
    light.location = center + Vector((0.0, -largest * 1.5, largest * 2.0))
    return cam, center, largest


def aim_camera(cam, center, largest, direction):
    direction = direction.normalized()
    cam.location = center + direction * largest * 3.0
    cam.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()


def render(path):
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", default=os.environ.get("PBR_CAPTURE_OBJ", ""))
    parser.add_argument("--texture", default=os.environ.get("PBR_CAPTURE_TEXTURE", ""))
    parser.add_argument("--out-dir", default=os.environ.get("PBR_CAPTURE_OUT_DIR", ""))
    parser.add_argument("--name", default=os.environ.get("PBR_CAPTURE_NAME", "capture"))
    parser.add_argument("--size", type=int, default=int(os.environ.get("PBR_CAPTURE_SIZE", "768")))
    parser.add_argument("--views", default=os.environ.get("PBR_CAPTURE_VIEWS", "front"))
    parser.add_argument(
        "--ortho-scale-mult",
        type=float,
        default=float(os.environ.get("PBR_CAPTURE_ORTHO_SCALE_MULT", "1.55")),
        help="orthographic camera scale multiplier relative to the largest object bound",
    )
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    if not args.obj or not args.texture or not args.out_dir:
        raise SystemExit("--obj, --texture, and --out-dir are required")

    clear_scene()
    objects = import_obj(Path(args.obj))
    cam, center, largest = setup_camera(objects, args.ortho_scale_mult)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 16
    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    views = parse_views(args.views)
    for view, direction in views:
        aim_camera(cam, center, largest, direction)
        prefix = args.name if len(views) == 1 else f"{args.name}_{view}"
        assign_color_material(objects, Path(args.texture))
        render(out_dir / f"{prefix}_color.png")
        assign_uv_material(objects)
        render(out_dir / f"{prefix}_uv.png")


if __name__ == "__main__":
    main()
