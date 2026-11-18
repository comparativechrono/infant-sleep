"""Headline and saving helpers for the annotated figures in figures/exploratory.

The annotated figures carry a headline, a subtitle and a data credit on the figure itself; they are
the versions first used to explore the results. The journal versions in figures/paper are drawn
separately, by paper_figures.py.
"""
import os
from pathlib import Path


def headline(fig, *args, **kwargs):
    """Figure-level headline, subtitle or data credit."""
    return fig.text(*args, **kwargs)


def save(fig, path, dpi, facecolor):
    """Write the figure to `path`, or into $FIG_DIR under the same file name when that is set."""
    path = Path(path)
    if os.environ.get("FIG_DIR"):
        path = Path(os.environ["FIG_DIR"]) / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=facecolor)
    return path
