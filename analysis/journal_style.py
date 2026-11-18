"""Shared style and helpers for the journal figures (paper_figures.py, letter_figures.py).

Figures are drawn at their printed size, 174 mm wide (the two-column width), so type sizes are the
printed sizes: Liberation Sans (metrically identical to Arial) at 7-8 pt with 10-pt bold panel
letters. Colours come from the Okabe-Ito set, which stays distinguishable with colour-vision
deficiency. Importing this module applies the matplotlib settings.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

MM = 1 / 25.4
WIDTH = 174 * MM                       # two-column width, inches
INK, INK_2 = "#000000", "#555555"
NA = "#c8c8c8"                         # not logged, or too little data to compute
BLUE, ORANGE = "#0072B2", "#D55E00"           # Okabe-Ito blue and vermilion
BAND = "#c6dcef"                       # interquartile bands around blue medians
NYU_LINE, NYU_BAND, NYU_BARS = "#666666", "#dddddd", "#b3b3b3"
RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", RAMP)
SEQ.set_bad(NA)
ACTO = LinearSegmentedColormap.from_list("asleep", ["#ffffff", "#000000"])  # share of bin asleep
ACTO.set_bad(NA)
T50 = r"$t_{50}$"
MONTH = 365.25 / 12

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
    "mathtext.cal": "Liberation Sans",
    "font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.pad": 2,
    "ytick.major.pad": 2, "axes.labelpad": 3, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5, "legend.borderaxespad": 0.3, "lines.linewidth": 1.0,
    "lines.solid_capstyle": "butt", "image.interpolation": "nearest", "pdf.fonttype": 42,
    "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def new_fig(height):
    return plt.figure(figsize=(WIDTH, height))


def ax_at(fig, x, y, w, h, **kw):
    """Axes placed in inches from the bottom-left corner of the figure."""
    W, H = fig.get_size_inches()
    return fig.add_axes((x / W, y / H, w / W, h / H), **kw)


def letter(fig, x, y, s, text=None):
    """Bold panel letter with its top-left corner at (x, y) inches; optional plain label after it."""
    W, H = fig.get_size_inches()
    fig.text(x / W, y / H, s, fontsize=10, fontweight="bold", ha="left", va="top")
    if text:
        fig.text((x + 0.17) / W, (y - 0.015) / H, text, fontsize=8, ha="left", va="top")


def clock(h):
    h = float(h) % 24
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


def noon_ref(h):
    """Clock hours (0-24) -> hours from the previous midnight-to-noon, so nights are continuous."""
    h = np.asarray(h, dtype=float)
    return np.where(h < 12, h + 24, h)


def signed(v, fmt="{:.2f}"):
    s = fmt.format(v + 0.0)
    return s.replace("-", "−")


def actogram_axis(ax):
    ax.set_xlim(0, 48)
    ax.set_xticks([0, 12, 24, 36, 48])
    ax.set_xticklabels(["0", "12", "24", "12", "24"])
    ax.set_xticks(np.arange(0, 49, 6), minor=True)



def save_figure(fig, out_dir, name):
    """Vector PDF (fonts embedded, no timestamps) and 300-dpi PNG."""
    fig.savefig(out_dir / f"{name}.pdf", dpi=600, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", out_dir / f"{name}.pdf")
