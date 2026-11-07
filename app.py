"""
Gradio web app for the low-light restoration system.
run locally:  python app.py   ->  http://localhost:7860
"""
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from src import metrics as M
from src.classical import enhance_classical
from src.hybrid import enhance_hybrid
from src.zerodce import enhance_zerodce

ROOT = Path(__file__).parent
MAX_SIDE = 1280          # bigger photos are resized so the app answers in a few seconds
EXAMPLES = sorted(str(p) for p in (ROOT / "samples").glob("*") if p.suffix.lower() in {".jpg", ".png", ".bmp"})


def to_float(img_u8):
    img = img_u8.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    if img.shape[2] == 4:
        img = img[..., :3]
    h, w = img.shape[:2]
    note = ""
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        note = f"Image was resized from {w}x{h} to {img.shape[1]}x{img.shape[0]} for speed. "
    return img, note


def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def heat(gray):
    return cv2.cvtColor(cv2.applyColorMap(to_u8(gray), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def run(image, method):
    if image is None:
        raise gr.Error("Please upload an image first.")
    if min(image.shape[:2]) < 32:
        raise gr.Error("Image is too small. Please use an image of at least 32x32 pixels.")

    img, note = to_float(image)
    t0 = time.perf_counter()
    gallery, info = [], {}
    if method == "Hybrid (recommended)":
        out, steps, info = enhance_hybrid(img)
        gallery = [(heat(steps["illumination"]), "1. Illumination map (classical)"),
                   (to_u8(steps["classical_branch"]), "2a. Retinex branch (classical)"),
                   (to_u8(steps["dl_branch"]), "2b. Zero-DCE branch (deep learning)"),
                   (heat(steps.get("fusion_weight_dl", np.zeros(img.shape[:2]))), "3. Fusion weight from illumination (bright = Zero-DCE, dark = Retinex)"),
                   (to_u8(steps["fused"]), "4. Fused"),
                   (to_u8(steps["denoised"]), "5. Denoised (bilateral)"),
                   (to_u8(steps["clahe"]), "6. CLAHE"),
                   (to_u8(steps["final"]), "7. Colour corrected (final)")]
    elif method == "Classical only":
        out, steps, info = enhance_classical(img)
        gallery = [(heat(steps["illumination"]), "1. Illumination map"),
                   (to_u8(steps["retinex"]), "2. Retinex"),
                   (to_u8(steps["denoised"]), "3. Denoised"),
                   (to_u8(steps["clahe"]), "4. CLAHE"),
                   (to_u8(steps["colour"]), "5. Colour corrected (final)")]
    else:
        out, info = enhance_zerodce(img)
        gallery = [(to_u8(out), f"Zero-DCE output ({info['chosen']} kept by quality check)")]
    ms = (time.perf_counter() - t0) * 1000

    rows = [["Processing time", f"{ms:.0f} ms"],
            ["NIQE input -> output (lower is better)", f"{M.niqe(img):.2f} -> {M.niqe(out):.2f}"],
            ["Mean brightness input -> output", f"{img.mean():.3f} -> {out.mean():.3f}"],
            ["Over-exposed pixels in output", f"{M.overexposed(out):.2f} %"]]
    for k in ("mean_lum", "dark_ratio", "noise_sigma", "colour_cast", "nonuniform",
              "gamma", "denoise_strength", "clahe_clip", "wb_strength", "dl_weight_share", "chosen"):
        if k in info:
            v = info[k]
            rows.append([k, f"{v:.3f}" if isinstance(v, float) else str(v)])

    status = note + f"Done with {method} in {ms:.0f} ms."
    return (image if not note else to_u8(img), to_u8(out)), gallery, rows, status


def compare(image):
    if image is None:
        raise gr.Error("Please upload an image first.")
    img, _ = to_float(image)
    outs = []
    for name, fn in [("Classical", lambda x: enhance_classical(x)[0]),
                     ("Deep learning (Zero-DCE)", lambda x: enhance_zerodce(x)[0]),
                     ("Hybrid", lambda x: enhance_hybrid(x)[0])]:
        t0 = time.perf_counter()
        o = fn(img)
        ms = (time.perf_counter() - t0) * 1000
        outs.append((to_u8(o), f"{name} | {ms:.0f} ms | NIQE {M.niqe(o):.2f}"))
    return outs


with gr.Blocks(title="Low-Light Image Restoration") as demo:
    gr.Markdown("# Low-Light Image Restoration\n"
                "Upload a dark, noisy photo. The hybrid method mixes a classical Retinex "
                "result with a Zero-DCE deep learning result, then denoises and fixes colour. "
                "All strengths are picked from the image itself.")
    with gr.Tab("Enhance"):
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Low-light image", type="numpy")
                method = gr.Radio(["Hybrid (recommended)", "Classical only", "Deep learning only"],
                                  value="Hybrid (recommended)", label="Method")
                btn = gr.Button("Enhance", variant="primary")
                status = gr.Markdown()
                if EXAMPLES:
                    gr.Examples(EXAMPLES, inputs=inp)
            with gr.Column(scale=2):
                slider = gr.ImageSlider(label="Before / after (drag the line)")
                table = gr.Dataframe(headers=["Item", "Value"], label="Image analysis and chosen parameters")
        steps = gr.Gallery(label="Intermediate steps", columns=4, height="auto")
        btn.click(run, [inp, method], [slider, steps, table, status])
    with gr.Tab("Compare all three"):
        with gr.Row():
            inp2 = gr.Image(label="Low-light image", type="numpy")
            btn2 = gr.Button("Run classical, deep learning and hybrid", variant="primary")
        out2 = gr.Gallery(label="Results", columns=3, height="auto")
        if EXAMPLES:
            gr.Examples(EXAMPLES, inputs=inp2)
        btn2.click(compare, inp2, out2)

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=7860)
