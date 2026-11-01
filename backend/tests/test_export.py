import importlib.util
import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest
import skia
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from export.blender import SCRIPT, find_blender
from export.svg import build_svg_zip, frame_to_svg
from pipeline.render import composite, render
from pipeline.strokes import Material, Region, Stroke, StrokeFrame

# The Blender script's pure helpers import fine outside Blender.
_spec = importlib.util.spec_from_file_location("build_grease_pencil", SCRIPT)
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)

SVG_NS = "{http://www.w3.org/2000/svg}"


def _frame() -> StrokeFrame:
    x = np.linspace(20, 180, 40, dtype=np.float32)
    wave = np.column_stack([x, 40 + 10 * np.sin(x / 20)]).astype(np.float32)
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    ring = np.column_stack([150 + 25 * np.cos(theta), 95 + 25 * np.sin(theta)]).astype(np.float32)
    square = np.array([[20, 70], [100, 70], [100, 130], [20, 130]], np.float32)
    hole = np.array([[45, 90], [75, 90], [75, 110], [45, 110]], np.float32)
    return StrokeFrame(
        width=200, height=140,
        materials=[Material("ink", "#17262b"), Material("fill_00", "#e06030", fill=True)],
        regions=[Region("fill_00", square, [hole])],
        strokes=[
            Stroke("ink", wave, np.full(40, 3, np.float32), np.ones(40, np.float32)),
            Stroke("ink", wave + [0, 15], np.linspace(1, 6, 40).astype(np.float32), np.full(40, 0.8, np.float32)),
            Stroke("ink", ring, np.linspace(2, 5, 60).astype(np.float32), np.ones(60, np.float32), closed=True),
        ],
    )


def _rasterise_svg(svg: str, w: int, h: int) -> np.ndarray:
    dom = skia.SVGDOM.MakeFromStream(skia.MemoryStream(svg.encode()))
    dom.setContainerSize(skia.Size(w, h))
    out = np.zeros((h, w, 4), np.uint8)
    surface = skia.Surface(out, colorType=skia.kRGBA_8888_ColorType)
    surface.getCanvas().clear(skia.ColorWHITE)
    dom.render(surface.getCanvas())
    surface.flushAndSubmit()
    return out[..., :3]


def test_svg_structure() -> None:
    frame = _frame()
    root = ET.fromstring(frame_to_svg(frame))
    assert root.get("viewBox") == "0 0 200 140"
    fills = root.find(f"{SVG_NS}g[@id='fills']")
    strokes = root.find(f"{SVG_NS}g[@id='strokes']")
    assert len(fills) == 1 and fills[0].get("fill-rule") == "evenodd"
    assert fills[0].get("d").count("M") == 2  # outline + hole
    uniform, variable, closed = list(strokes)
    assert uniform.get("stroke-width") == "3" and uniform.get("fill") == "none"
    assert variable.get("fill") == "#17262b" and variable.get("fill-opacity") == "0.800"
    assert closed.get("fill-rule") == "evenodd"


def test_svg_renders_like_the_preview() -> None:
    frame = _frame()
    svg = _rasterise_svg(frame_to_svg(frame), frame.width, frame.height).astype(int)
    preview = composite(render(frame), np.full((140, 200, 3), 255, np.uint8)).astype(int)
    diff = np.abs(svg - preview).mean(-1)
    assert diff.mean() < 3.0  # only antialiasing differences
    assert (svg[100, 60] == 255).all()  # the hole shows the background


def test_svg_zip(tmp_path: Path) -> None:
    strokes = tmp_path / "strokes"
    strokes.mkdir()
    for i in range(3):
        (strokes / f"{i:05d}.json").write_text(json.dumps(_frame().to_json()))
    out = build_svg_zip(strokes, tmp_path / "svg.zip")
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == [f"frames/frame_0000{i}.svg" for i in range(3)]
        ET.fromstring(zf.read("frames/frame_00001.svg"))


def test_blender_helpers() -> None:
    assert gp.fps_fraction(24) == (24, 1.0)
    assert gp.fps_fraction(29.97) == (30, 1.001)
    assert gp.fps_fraction(14.985) == (15, 1.001)  # half of 29.97
    fps, base = gp.fps_fraction(12.5)
    assert abs(fps / base - 12.5) < 1e-6

    s = gp.scale_for(1280, 720)
    assert s == pytest.approx(4.0 / 1280)
    (x0, y0, z0), (x1, _, z1) = gp.to_blender([(0, 0), (1280, 720)], 1280, 720, s)
    assert (x0, y0, z0) == pytest.approx((-2.0, 0.0, 1.125)) and (x1, z1) == pytest.approx((2.0, -1.125))

    assert gp.hex_to_rgba("#ffffff") == pytest.approx((1, 1, 1, 1))
    assert gp.hex_to_rgba("#808080")[0] == pytest.approx(0.2159, abs=1e-3)  # sRGB -> linear
    assert gp.blender_opacity(1.0) == 1.0 and gp.blender_opacity(0.0) == 0.0
    assert gp.blender_opacity(0.5) == pytest.approx(0.786, abs=1e-3)

    widths = [2.0, 4.0, 3.0]
    line_width, pressures = gp.gpv2_thickness(widths, s)
    radii = [lw * line_width * gp.GPV2_THICKNESS_FACTOR for lw in pressures]
    assert radii == pytest.approx([w / 2 * s for w in widths])


def test_frame_items_order_and_holes() -> None:
    fills, lines = gp.frame_items(_frame().to_json(), gp.scale_for(200, 140))
    assert [f["material"] for f in fills] == ["fill_00", "fill_hole"]
    assert all(f["cyclic"] for f in fills)
    assert [line["cyclic"] for line in lines] == [False, False, True]
    assert lines[1]["radius"][0] == pytest.approx(0.5 * gp.scale_for(200, 140))


BLENDER = find_blender(None)


@pytest.mark.skipif(not BLENDER.available, reason="Blender not installed")
def test_blend_round_trip(tmp_path: Path) -> None:
    from export.blender import export_blend

    strokes = tmp_path / "strokes"
    strokes.mkdir()
    for i in range(3):
        (strokes / f"{i:05d}.json").write_text(json.dumps(_frame().to_json()))
    out = export_blend(BLENDER, strokes, tmp_path / "out.blend", fps=29.97)
    check = tmp_path / "check.py"
    check.write_text(
        "import bpy, json\n"
        "o = [o for o in bpy.data.objects if o.type in ('GREASEPENCIL', 'GPENCIL')][0]\n"
        "s = bpy.context.scene\n"
        "def counts(layer):\n"
        "    f = layer.frames[0]\n"
        "    return len(f.drawing.strokes) if hasattr(f, 'drawing') else len(f.strokes)\n"
        "print('RESULT ' + json.dumps({'layers': [l.name for l in o.data.layers],"
        " 'frames': [len(l.frames) for l in o.data.layers], 'strokes': [counts(l) for l in o.data.layers],"
        " 'materials': sorted(m.name for m in o.data.materials), 'fps': [s.render.fps, s.render.fps_base],"
        " 'res': [s.render.resolution_x, s.render.resolution_y], 'end': s.frame_end,"
        " 'camera': s.camera.data.type}))\n"
    )
    res = subprocess.run([BLENDER.path, "-b", str(out), "-P", str(check)], capture_output=True, text=True, timeout=300)
    line = next(ln for ln in res.stdout.splitlines() if ln.startswith("RESULT "))
    info = json.loads(line[7:])
    assert info["layers"] == ["Fills", "Lines"]
    assert info["frames"] == [3, 3]
    assert info["strokes"] == [2, 3]  # fill + hole holdout; three line strokes
    assert info["materials"] == ["fill_00", "fill_hole", "ink"]
    assert info["fps"][0] == 30 and info["fps"][1] == pytest.approx(1.001)
    assert info["res"] == [200, 140] and info["end"] == 3 and info["camera"] == "ORTHO"


@pytest.fixture()
def render_client(tmp_path: Path, sample_clip: Path):
    """A project with one finished (pencil, no models needed) render."""
    settings = Settings(storage_dir=tmp_path / "storage", prefetch_models=False, blender_path="/nonexistent/blender")
    with TestClient(create_app(settings)) as c:
        up = c.post("/api/uploads", params={"filename": "clip.mp4"}, content=sample_clip.read_bytes()).json()
        pid = c.post(f"/api/uploads/{up['upload_id']}/confirm").json()["id"]
        rid = "ab" * 16
        d = tmp_path / "storage" / "projects" / pid / "renders" / rid
        (d / "strokes").mkdir(parents=True)
        for i in range(2):
            (d / "strokes" / f"{i:05d}.json").write_text(json.dumps(_frame().to_json()))
        (d / "render.json").write_text(json.dumps({"id": rid, "style": "pencil", "fps": 12.0}))
        yield c, pid, rid


def test_export_api(render_client) -> None:
    c, pid, rid = render_client
    base = f"/api/projects/{pid}/renders/{rid}/exports"
    assert c.get(f"{base}/svg").status_code == 404  # not built yet
    res = c.post(f"{base}/svg")
    assert res.status_code == 200 and res.json()["ready"]
    dl = c.get(f"{base}/svg")
    assert dl.status_code == 200 and dl.headers["content-type"] == "application/zip"
    assert 'filename="clip-pencil-svg_frames.zip"' in dl.headers["content-disposition"]

    # Blender is configured to a path that doesn't exist: clear error, export disabled.
    res = c.post(f"{base}/blend")
    assert res.status_code == 503 and "BLENDER_PATH" in res.json()["detail"]
    system = c.get("/api/system").json()["blender"]
    assert system["available"] is False and "BLENDER_PATH" in system["error"]
