"""Switch between the annotated figures and the journal versions.

By default each figure script writes the annotated version (headline, subtitle and data credit
drawn on the figure) into its output folder, or into $FIG_DIR when that is set.
With PAPER_FIGURES=<folder> in the environment the same script writes the journal version
instead: no figure-level headline, subtitle or credit (the legend carries them), panel letters
in front of the panel titles, and a 300-dpi PNG plus a vector PDF, cropped to the drawing.
"""
import os
from pathlib import Path

PAPER_DIR = os.environ.get("PAPER_FIGURES")
PAPER = bool(PAPER_DIR)
_panels = []


def headline(fig, *args, **kwargs):
    """Figure-level headline, subtitle or data credit: drawn in the annotated version only."""
    if not PAPER:
        return fig.text(*args, **kwargs)
    return None


def lettered(text):
    """Panel title; in the journal version the next panel letter goes in front."""
    if not PAPER:
        return text
    _panels.append(text)
    return f"{chr(64 + len(_panels))}   {text}"


def save(fig, path, dpi, paper_name, facecolor):
    """Write the figure and return the main file written. Panel letters restart after each save."""
    _panels.clear()
    if PAPER:
        out = Path(PAPER_DIR)
        out.mkdir(parents=True, exist_ok=True)
        for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
            fig.savefig(out / f"{paper_name}.{ext}", bbox_inches="tight", pad_inches=0.12,
                        facecolor=facecolor, **kw)
        return out / f"{paper_name}.png"
    path = Path(path)
    if os.environ.get("FIG_DIR"):
        path = Path(os.environ["FIG_DIR"]) / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=facecolor)
    return path
