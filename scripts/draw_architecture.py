"""Draws docs/architecture.png (system architecture diagram)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, GREY = "#dbe8f7", "#fde3c8", "#d9efd9", "#eeeeee"

fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.2)
ax.axis("off")


def box(x, y, w, h, text, color, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="#555555", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2,
            fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2, label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2))
    if label:
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2 + 0.08, label, fontsize=8, color="#444444")


ax.text(6.5, 6.95, "Low-light restoration system (Problem P1): hybrid pipeline", ha="center",
        fontsize=13, fontweight="bold")

box(0.2, 3.1, 1.6, 0.9, "Input image\n(upload in app)", GREY, True)
box(2.2, 4.9, 2.2, 1.1, "Image analysis\nbrightness, dark %, noise,\ncolour cast, contrast", BLUE)
box(2.2, 3.1, 2.2, 1.1, "Illumination map T\nmax(R,G,B) +\nguided filter", BLUE)
box(2.2, 1.3, 2.2, 1.1, "Pre-gain\n(only if very dark)\nRetinex lift to 0.2", BLUE)

box(5.0, 4.1, 2.2, 1.1, "Branch A: Retinex\nI x T^(gamma-1)\ngamma from target", BLUE)
box(5.0, 1.8, 2.2, 1.3, "Branch B: Zero-DCE\n8 curve maps (79k params)\nfine-tuned, 512px curves\n+ quality check 4 vs 8", ORANGE)

box(7.8, 2.9, 2.0, 1.3, "Adaptive fusion\nper-pixel weight\nfrom T: dark -> Retinex\nlit -> Zero-DCE", GREEN, True)

box(10.3, 5.0, 2.4, 0.8, "Bilateral denoise (YCrCb)\nmore in dark areas of T", BLUE)
box(10.3, 3.85, 2.4, 0.8, "CLAHE on L channel\n(clip from noise/contrast)", BLUE)
box(10.3, 2.7, 2.4, 0.8, "White balance (capped)\n+ highlight protection", BLUE)
box(10.3, 1.3, 2.4, 0.9, "Output + intermediate\nsteps + NIQE, latency", GREY, True)

arrow(1.8, 3.55, 2.2, 3.55)
arrow(1.8, 3.7, 2.2, 5.3)
arrow(3.3, 3.1, 3.3, 2.4)
arrow(4.4, 3.65, 5.0, 4.5)
ax.annotate('', xy=(7.8, 3.3), xytext=(4.4, 3.4), arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.2, ls='--'))
ax.text(5.15, 3.5, 'T used as fusion weight', fontsize=8, color='#2e7d32')
arrow(4.4, 1.85, 5.0, 2.3)
arrow(4.4, 5.3, 5.0, 4.9)
ax.text(4.5, 5.55, "parameters", fontsize=8, color="#444444")
arrow(7.2, 4.6, 7.8, 3.9)
arrow(7.2, 2.5, 7.8, 3.2)
arrow(9.8, 3.55, 10.3, 5.3)
arrow(11.5, 5.0, 11.5, 4.65)
arrow(11.5, 3.85, 11.5, 3.5)
arrow(11.5, 2.7, 11.5, 2.2)
ax.annotate("", xy=(10.3, 5.6), xytext=(3.3, 6.0),
            arrowprops=dict(arrowstyle="->", color="#999999", lw=1, ls="--",
                            connectionstyle="arc3,rad=-0.12"))
ax.text(6.6, 6.35, "noise sigma, gain, cast -> denoise / CLAHE / WB strength", fontsize=8, color="#666666")

ax.add_patch(FancyBboxPatch((0.2, 0.15), 12.5, 0.75, boxstyle="round,pad=0.02", fc="white", ec="#bbbbbb"))
ax.text(0.4, 0.52, "Blue = classical CV (OpenCV / NumPy)      Orange = deep learning (PyTorch)      "
        "Green = hybrid decision      Deployed as a Gradio app (Hugging Face Spaces / Docker)",
        fontsize=8.6, va="center")

(ROOT / "docs").mkdir(exist_ok=True)
fig.savefig(ROOT / "docs/architecture.png", dpi=150, bbox_inches="tight")
print("saved docs/architecture.png")
