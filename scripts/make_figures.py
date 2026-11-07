"""
Makes the figures used in the README and report (after evaluate.py has run):
  docs/compare_real.jpg      real dark photos: input | classical | DL | hybrid
  docs/compare_synthetic.jpg synthetic test: input | classical | DL | hybrid | ground truth
  docs/failures.jpg          the cases where the hybrid does worst
  docs/intermediate.jpg      every step of the hybrid pipeline for one photo
  docs/results_chart.png     PSNR / SSIM per lighting condition
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hybrid import enhance_hybrid  # noqa: E402
from src.classical import enhance_classical  # noqa: E402
from src.zerodce import enhance_zerodce  # noqa: E402
from src.io_utils import read_rgb  # noqa: E402

DOCS = ROOT / "docs"
W, H = 300, 220


def tile(img, label):
    t = cv2.resize((np.clip(img, 0, 1) * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA)
    t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
    cv2.rectangle(t, (0, 0), (W, 22), (0, 0, 0), -1)
    cv2.putText(t, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def methods(img):
    return [("classical", enhance_classical(img)[0]),
            ("Zero-DCE (fine-tuned)", enhance_zerodce(img)[0]),
            ("hybrid (ours)", enhance_hybrid(img)[0])]


def real_grid(names):
    rows = []
    for p in names:
        img = read_rgb(ROOT / p, 800)
        rows.append(np.hstack([tile(img, f"input {Path(p).parent.name}/{Path(p).name}")] +
                              [tile(o, n) for n, o in methods(img)]))
    cv2.imwrite(str(DOCS / "compare_real.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def synth_grid(items, out_name, title_extra=None):
    rows = []
    for k, (cond, name) in enumerate(items):
        low = read_rgb(ROOT / "data/synth/test" / cond / "low" / name)
        gt = read_rgb(ROOT / "data/synth/test" / cond / "high" / name)
        extra = f" {title_extra[k]}" if title_extra else ""
        rows.append(np.hstack([tile(low, f"input ({cond}){extra}")] +
                              [tile(o, n) for n, o in methods(low)] + [tile(gt, "ground truth")]))
    cv2.imwrite(str(DOCS / out_name), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def intermediate(path):
    img = read_rgb(ROOT / path, 800)
    _, s, info = enhance_hybrid(img)
    heat = lambda g: cv2.cvtColor(cv2.applyColorMap((np.clip(g, 0, 1) * 255).astype(np.uint8),
                                                   cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB) / 255.0
    tiles = [tile(img, "0 input"), tile(heat(s["illumination"]), "1 illumination T"),
             tile(s["classical_branch"], "2a Retinex branch"), tile(s["dl_branch"], "2b Zero-DCE branch"),
             tile(heat(s["fusion_weight_dl"]), "3 DL weight (from T)"), tile(s["fused"], "4 fused"),
             tile(s["denoised"], "5 denoised"), tile(s["final"], "6 CLAHE + colour = final")]
    grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    cv2.imwrite(str(DOCS / "intermediate.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


def chart():
    df = pd.read_csv(ROOT / "results/paired_synth_by_condition.csv")
    df = df[df["method"] != "input (no enhancement)"]
    conds = ["dark", "extreme", "nonuniform"]
    meths = list(dict.fromkeys(df["method"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    colors = ["#4c72b0", "#c9a227", "#dd8452", "#2e8b57"]
    for ax, metric in zip(axes, ["PSNR", "SSIM"]):
        x = np.arange(len(conds))
        width = 0.8 / len(meths)
        for i, m in enumerate(meths):
            vals = [df[(df.condition == c) & (df.method == m)][metric].values[0] for c in conds]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=m, color=colors[i % len(colors)])
        ax.set_xticks(x)
        ax.set_xticklabels(conds)
        ax.set_title(f"{metric} on unseen test set (higher is better)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("dB")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(DOCS / "results_chart.png", dpi=140)


def failures():
    df = pd.read_csv(ROOT / "results/paired_synth_per_image.csv")
    hy = df[df.method == "hybrid (ours)"].sort_values("PSNR")
    # worst image per lighting condition, three different photos
    picked, seen = [], set()
    for cond in ("dark", "extreme", "nonuniform"):
        row = hy[(hy.condition == cond) & (~hy.image.isin(seen))].iloc[0]
        picked.append(row)
        seen.add(row.image)
    hy = pd.DataFrame(picked)
    synth_grid(list(zip(hy.condition, hy.image)), "failures.jpg",
               [f"PSNR {v:.1f}" for v in hy.PSNR])
    return hy[["condition", "image", "PSNR", "SSIM", "DeltaE"]]


def chart_official(out="docs/results_chart_lol.png"):
    """PSNR / SSIM on the official test sets (needs results/paired_<set>_summary.csv)."""
    sets = [("lolv2_real", "LOL-v2 Real (100)"), ("lolv2_synthetic", "LOL-v2 Synthetic (100)"),
            ("lol_v1_eval15", "LOL v1 eval15 (15)")]
    sets = [(k, n) for k, n in sets if (ROOT / f"results/paired_{k}_summary.csv").exists()]
    meths = ["classical", "DL Zero-DCE pretrained", "DL Zero-DCE fine-tuned", "hybrid (ours)"]
    colors = ["#4c72b0", "#c9a227", "#dd8452", "#2e8b57"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, metric in zip(axes, ["PSNR", "SSIM"]):
        x = np.arange(len(sets))
        width = 0.8 / len(meths)
        for i, m in enumerate(meths):
            vals = [pd.read_csv(ROOT / f"results/paired_{k}_summary.csv", index_col=0).loc[m, metric] for k, _ in sets]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=m, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in sets], fontsize=9)
        ax.set_title(f"{metric} on official test sets (higher is better)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("dB")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(ROOT / out, dpi=140)


def lol_grid(test_dir, out_name="compare_lolv2.jpg", n=4):
    """official LOL-v2 Real test: darkest / brightest / non-uniform examples + ground truth.
    Also writes the hybrid's worst PSNR images to docs/failures_lolv2.jpg."""
    from src.datasets import find_pairs
    from src import metrics as M
    pairs = find_pairs(test_dir)
    lum = [(read_rgb(a, 600).mean(), a, b) for a, b in pairs]
    lum.sort(key=lambda r: r[0])
    # 2 darkest (extreme low light), the most non-uniform one, and a median one
    picks = [lum[0], lum[1]]
    def spread(a):
        y = read_rgb(a, 300).mean(2)
        b = cv2.GaussianBlur(y, (0, 0), 15)
        return b.std() / (b.mean() + 1e-3)
    rest = [r for r in lum[2:]]
    picks.append(max(rest, key=lambda r: spread(r[1])))
    picks.append(lum[len(lum) // 2])
    rows, scores = [], []
    for k, (m, a, b) in enumerate(picks[:n]):
        low, gt = read_rgb(a, 600), read_rgb(b, 600)
        tag = ["extreme dark", "extreme dark", "non-uniform", "typical"][k]
        rows.append(np.hstack([tile(low, f"input {a.name} ({tag})")] +
                              [tile(o, nm) for nm, o in methods(low)] + [tile(gt, "ground truth")]))
    cv2.imwrite(str(DOCS / out_name), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    for a, b in pairs:
        low, gt = read_rgb(a, 600), read_rgb(b, 600)
        scores.append((M.psnr(enhance_hybrid(low)[0], gt), a, b))
    scores.sort(key=lambda r: r[0])
    rows = []
    for ps, a, b in scores[:3]:
        low, gt = read_rgb(a, 600), read_rgb(b, 600)
        rows.append(np.hstack([tile(low, f"input {a.name}")] + [tile(o, nm) for nm, o in methods(low)] +
                              [tile(gt, f"ground truth (hybrid PSNR {ps:.1f})")]))
    cv2.imwrite(str(DOCS / "failures_lolv2.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return [(f"{a.name}", round(ps, 2)) for ps, a, _ in scores[:3]]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lol_test", default="LOL-v2/Real_captured/Test",
                    help="official test folder used for docs/compare_lolv2.jpg and failures_lolv2.jpg")
    args = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    real_grid(["data/raw/real_dark/LIME/1.bmp", "data/raw/real_dark/LIME/7.bmp",
               "data/raw/real_dark/DICM/03.jpg", "data/raw/real_dark/RetinexNet/eval_22.png"])
    intermediate("data/raw/real_dark/LIME/1.bmp")
    chart_official()
    if (ROOT / args.lol_test).exists():
        print("hybrid worst cases on LOL-v2 Real test:", lol_grid(ROOT / args.lol_test))
    else:
        print(f"{args.lol_test} not found, skipping LOL figures")
    if (ROOT / "data/synth/test").exists():      # our earlier darkened-photo experiments
        synth_grid([("dark", "Kodak24_kodim05.png"), ("extreme", "Kodak24_kodim23.png"),
                    ("nonuniform", "McMaster_10.png")], "compare_synthetic.jpg")
        chart()
        print(failures().to_string(index=False))
    print("figures written to docs/")
