from . import clean, pencil, rotoscope
from .base import FrameInput, Style, StyleParams

STYLES: dict[str, Style] = {s.key: s for s in (clean.STYLE, pencil.STYLE, rotoscope.STYLE)}

__all__ = ["STYLES", "FrameInput", "Style", "StyleParams"]
