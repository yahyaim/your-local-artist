"""Build a Grease Pencil .blend from my-local-artist stroke JSON. Runs inside Blender, headless:

    blender -b --factory-startup -P blender_export/build_grease_pencil.py -- \\
        --strokes <render>/strokes --output out.blend [--fps 24] [--name Strokes]

Target: Blender 4.3+ ("Grease Pencil v3", the rewritten GP), developed and tested
against Blender 4.5 LTS. Blender 4.0-4.2 (legacy "GPv2" API) is handled by a
separate, simpler code path that has not been tested on a real 4.0-4.2 install.

What you get:
- One Grease Pencil object with two layers, "Fills" (rotoscope colour regions)
  under "Lines" (all strokes), one keyframe per video frame starting at frame 1.
- One material per stroke material in the JSON ("ink", "graphite", "fill_00"...),
  plus "fill_hole" (a fill holdout that cuts holes, e.g. the gap between arm and body).
- Scene fps, resolution and frame range matching the video, an orthographic
  camera that frames the drawing exactly like the video frame, white world and
  the Standard view transform (colours match the preview video).

Coordinates: the drawing lies in the XZ plane (Grease Pencil's default "front"
plane), centred on the origin, with the frame's long side 4 m wide.
"""

import argparse
import json
import math
import sys
from pathlib import Path

LONG_SIDE_M = 4.0
# Legacy GPv2 stores thickness as integer "pixels"; Blender's own GPv2 -> GPv3
# conversion uses radius = line_width * pressure / 2000.
GPV2_THICKNESS_FACTOR = 1.0 / 2000.0


# --- Pure helpers (no bpy; covered by backend/tests/test_export.py) ---------------


def scale_for(width: int, height: int) -> float:
    """Metres per pixel."""
    return LONG_SIDE_M / max(width, height)


def to_blender(points, width: int, height: int, scale: float) -> list[tuple[float, float, float]]:
    """Pixel (x right, y down) -> Blender XZ plane (x right, z up), centred."""
    return [((x - width / 2) * scale, 0.0, (height / 2 - y) * scale) for x, y in points]


def fps_fraction(fps: float) -> tuple[int, float]:
    """Blender stores fps as an integer plus a divisor: 29.97 -> (30, 1.001)."""
    if abs(fps - round(fps)) < 0.01:
        return int(round(fps)), 1.0
    ntsc = round(fps * 1.001)
    if abs(ntsc / 1.001 - fps) < 0.01:
        return int(ntsc), 1.001
    # Arbitrary rates (e.g. half of 29.97): keep the integer part exact via the base.
    return int(math.ceil(fps)), math.ceil(fps) / fps


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_to_rgba(color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """sRGB hex -> linear RGBA, which is what Blender material colours expect."""
    h = color.lstrip("#")
    return (*(srgb_to_linear(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4)), alpha)


def blender_opacity(opacity: float) -> float:
    """Stroke opacity that looks the same in Blender as in the preview.

    The preview (Skia) blends in sRGB; Blender blends in linear light, where 50%
    black over white shows as ~27% grey. For dark ink on a light background (all
    our styles) this makes Blender match: blend amount such that the linear
    result displays as the same sRGB grey.
    """
    return 1.0 - srgb_to_linear(1.0 - opacity)


def gpv2_thickness(widths_px: list[float], scale: float) -> tuple[int, list[float]]:
    """Per-point pixel widths -> (stroke line_width, per-point pressure) for legacy GPv2."""
    radii = [w / 2 * scale for w in widths_px]
    line_width = max(1, math.ceil(max(radii) / GPV2_THICKNESS_FACTOR))
    return line_width, [r / (line_width * GPV2_THICKNESS_FACTOR) for r in radii]


def frame_items(frame: dict, scale: float) -> tuple[list[dict], list[dict]]:
    """A stroke JSON frame -> (fill items, line items) in drawing order.

    Each item: {material, points (blender xyz), radius (m, per point), opacity, cyclic}.
    Fills are painted largest first; each region's holes come right after it as
    "fill_hole" holdout strokes (everything inside a hole that should be visible
    is a smaller region, so it is painted later, on top).
    """
    w, h = frame["width"], frame["height"]
    fills, lines = [], []
    for region in frame["regions"]:
        for pts, material in [(region["outline"], region["material"])] + [(hole, "fill_hole") for hole in region["holes"]]:
            if len(pts) >= 3:
                fills.append({
                    "material": material, "points": to_blender(pts, w, h, scale),
                    "radius": [0.0005] * len(pts), "opacity": [1.0] * len(pts), "cyclic": True,
                })
    for stroke in frame["strokes"]:
        pts = stroke["points"]
        if len(pts) < 2:
            continue
        lines.append({
            "material": stroke["material"],
            "points": to_blender([(p[0], p[1]) for p in pts], w, h, scale),
            "radius": [p[2] / 2 * scale for p in pts],
            "opacity": [blender_opacity(p[3]) for p in pts],
            "cyclic": stroke["closed"],
        })
    return fills, lines


# --- Blender side ------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Grease Pencil .blend from stroke JSON")
    parser.add_argument("--strokes", required=True, type=Path, help="folder with 00000.json, 00001.json, ...")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--name", default="Strokes")
    return parser.parse_args(argv)


def main() -> None:
    import bpy  # only available inside Blender

    args = _parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    files = sorted(args.strokes.glob("*.json"))
    if not files:
        sys.exit(f"No stroke JSON files in {args.strokes}")
    first = json.loads(files[0].read_text())
    width, height = first["width"], first["height"]
    scale = scale_for(width, height)

    # Empty scene: factory startup comes with a cube, light and camera.
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene = bpy.context.scene

    # Blender 4.3+ has the new Grease Pencil ("v3"); 4.3-4.5 expose it as
    # bpy.data.grease_pencils_v3, 5.0+ as bpy.data.grease_pencils.
    if hasattr(bpy.data, "grease_pencils_v3"):
        builder = _GPv3(bpy, bpy.data.grease_pencils_v3, args.name)
    elif bpy.app.version >= (5, 0, 0):
        builder = _GPv3(bpy, bpy.data.grease_pencils, args.name)
    else:
        builder = _GPv2(bpy, args.name)
    scene.collection.objects.link(builder.obj)
    print(f"Blender {bpy.app.version_string}: using {builder.__class__.__name__.strip('_')} API")

    for i, path in enumerate(files):
        frame = first if i == 0 else json.loads(path.read_text())
        builder.add_frame(i + 1, frame, scale)
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            print(f"PROGRESS {i + 1}/{len(files)}", flush=True)

    # Scene: fps, resolution, range, and a camera that frames the video exactly.
    scene.render.fps, scene.render.fps_base = fps_fraction(args.fps)
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.frame_start, scene.frame_end, scene.frame_current = 1, len(files), 1
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(width, height) * scale
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = (0.0, -10.0, 0.0)
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)
    scene.collection.objects.link(cam)
    scene.camera = cam
    # White background, and the "Standard" view transform so material colours
    # render as the same sRGB values as the preview video (Blender's default AgX
    # transform would mute them and turn white grey).
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (1.0, 1.0, 1.0)
    if world.use_nodes and "Background" in world.node_tree.nodes:
        world.node_tree.nodes["Background"].inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()), compress=True)
    print(f"Saved {args.output} ({len(files)} frames, {width}x{height}, {args.fps} fps)")


class _Materials:
    """Creates Grease Pencil materials on first use and maps names to slot indices."""

    def __init__(self, bpy, obj) -> None:
        self.bpy, self.obj, self.index = bpy, obj, {}

    def slot(self, name: str, frame: dict) -> int:
        if name in self.index:
            return self.index[name]
        bpy = self.bpy
        mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        if mat.grease_pencil is None:
            bpy.data.materials.create_gpencil_data(mat)
        gp = mat.grease_pencil
        if name == "fill_hole":
            gp.show_stroke, gp.show_fill, gp.use_fill_holdout = False, True, True
        else:
            spec = next((m for m in frame["materials"] if m["name"] == name), {"color": "#000000", "fill": False})
            if spec["fill"]:
                gp.show_stroke, gp.show_fill = False, True
                gp.fill_color = hex_to_rgba(spec["color"])
            else:
                gp.show_stroke, gp.show_fill = True, False
                gp.color = hex_to_rgba(spec["color"])
        self.obj.data.materials.append(mat)
        self.index[name] = len(self.obj.data.materials) - 1
        return self.index[name]


class _GPv3:
    """Blender 4.3+ Grease Pencil: bulk attribute writes per drawing (fast)."""

    def __init__(self, bpy, collection, name: str) -> None:
        self.data = collection.new(name)
        self.obj = bpy.data.objects.new(name, self.data)
        self.materials = _Materials(bpy, self.obj)
        self.fills = self.data.layers.new("Fills")
        self.lines = self.data.layers.new("Lines")  # created last = drawn on top

    @staticmethod
    def _attr(drawing, name: str, kind: str, domain: str):
        return drawing.attributes.get(name) or drawing.attributes.new(name, kind, domain)

    def _write(self, layer, number: int, items: list[dict], frame: dict) -> None:
        drawing = layer.frames.new(number).drawing
        if not items:
            return
        drawing.add_strokes([len(it["points"]) for it in items])
        flat = [c for it in items for p in it["points"] for c in p]
        drawing.attributes["position"].data.foreach_set("vector", flat)
        self._attr(drawing, "radius", "FLOAT", "POINT").data.foreach_set(
            "value", [r for it in items for r in it["radius"]])
        self._attr(drawing, "opacity", "FLOAT", "POINT").data.foreach_set(
            "value", [o for it in items for o in it["opacity"]])
        self._attr(drawing, "material_index", "INT", "CURVE").data.foreach_set(
            "value", [self.materials.slot(it["material"], frame) for it in items])
        self._attr(drawing, "cyclic", "BOOLEAN", "CURVE").data.foreach_set(
            "value", [it["cyclic"] for it in items])

    def add_frame(self, number: int, frame: dict, scale: float) -> None:
        fills, lines = frame_items(frame, scale)
        self._write(self.fills, number, fills, frame)
        self._write(self.lines, number, lines, frame)


class _GPv2:
    """Blender 4.0-4.2 legacy Grease Pencil (untested on a real install; see module docstring)."""

    def __init__(self, bpy, name: str) -> None:
        self.data = bpy.data.grease_pencils.new(name)
        self.data.stroke_depth_order = "2D"
        self.obj = bpy.data.objects.new(name, self.data)
        self.materials = _Materials(bpy, self.obj)
        self.fills = self.data.layers.new("Fills", set_active=False)
        self.lines = self.data.layers.new("Lines", set_active=True)

    def _write(self, layer, number: int, items: list[dict], frame: dict, scale: float) -> None:
        gp_frame = layer.frames.new(number)
        for it in items:
            stroke = gp_frame.strokes.new()
            stroke.display_mode = "3DSPACE"
            stroke.use_cyclic = it["cyclic"]
            stroke.material_index = self.materials.slot(it["material"], frame)
            widths_px = [r * 2 / scale for r in it["radius"]]
            stroke.line_width, pressures = gpv2_thickness(widths_px, scale)
            stroke.points.add(len(it["points"]))
            stroke.points.foreach_set("co", [c for p in it["points"] for c in p])
            stroke.points.foreach_set("pressure", pressures)
            stroke.points.foreach_set("strength", it["opacity"])

    def add_frame(self, number: int, frame: dict, scale: float) -> None:
        fills, lines = frame_items(frame, scale)
        self._write(self.fills, number, fills, frame, scale)
        self._write(self.lines, number, lines, frame, scale)


if __name__ == "__main__":
    main()
