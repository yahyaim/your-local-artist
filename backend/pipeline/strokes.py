"""Vector stroke data: the single source of truth for preview, SVG and Blender.

One `StrokeFrame` per video frame, serialised as JSON:

    {
      "format": "my-local-artist/strokes", "version": 1,
      "width": 1280, "height": 720,
      "materials": [{"name": "ink", "color": "#17262b", "fill": false}, ...],
      "regions": [{"material": "fill_0", "outline": [[x, y], ...], "holes": [[[x, y], ...]]}],
      "strokes": [{"material": "ink", "closed": false, "points": [[x, y, width, opacity], ...]}]
    }

Coordinates are pixels, origin top-left, y down. `width` is the full line width
in pixels. Regions (flat colour fills) are painted first, largest first, then
strokes on top in list order.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

FORMAT = "my-local-artist/strokes"
VERSION = 1


@dataclass
class Material:
    name: str
    color: str  # "#rrggbb"
    fill: bool = False  # True for region fills, False for line strokes


@dataclass
class Stroke:
    material: str
    points: np.ndarray  # (N, 2) float32
    widths: np.ndarray  # (N,) float32
    opacities: np.ndarray  # (N,) float32 in [0, 1]
    closed: bool = False


@dataclass
class Region:
    material: str
    outline: np.ndarray  # (N, 2) float32, implicitly closed
    holes: list[np.ndarray] = field(default_factory=list)


@dataclass
class StrokeFrame:
    width: int
    height: int
    materials: list[Material]
    regions: list[Region] = field(default_factory=list)
    strokes: list[Stroke] = field(default_factory=list)

    def material(self, name: str) -> Material:
        return next(m for m in self.materials if m.name == name)

    def to_json(self) -> dict[str, Any]:
        def pts(a: np.ndarray) -> list:
            return np.round(a, 2).tolist()

        return {
            "format": FORMAT,
            "version": VERSION,
            "width": self.width,
            "height": self.height,
            "materials": [{"name": m.name, "color": m.color, "fill": m.fill} for m in self.materials],
            "regions": [
                {"material": r.material, "outline": pts(r.outline), "holes": [pts(h) for h in r.holes]}
                for r in self.regions
            ],
            "strokes": [
                {
                    "material": s.material,
                    "closed": s.closed,
                    "points": np.column_stack(
                        [np.round(s.points, 2), np.round(s.widths, 2), np.round(s.opacities, 3)]
                    ).tolist(),
                }
                for s in self.strokes
            ],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "StrokeFrame":
        if data.get("format") != FORMAT:
            raise ValueError("not a stroke frame file")

        def arr(v: list) -> np.ndarray:
            return np.asarray(v, dtype=np.float32).reshape(-1, 2)

        strokes = []
        for s in data["strokes"]:
            p = np.asarray(s["points"], dtype=np.float32).reshape(-1, 4)
            strokes.append(Stroke(s["material"], p[:, :2].copy(), p[:, 2].copy(), p[:, 3].copy(), s["closed"]))
        return cls(
            width=data["width"],
            height=data["height"],
            materials=[Material(**m) for m in data["materials"]],
            regions=[Region(r["material"], arr(r["outline"]), [arr(h) for h in r["holes"]]) for r in data["regions"]],
            strokes=strokes,
        )
