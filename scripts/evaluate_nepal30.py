import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import build_model

PROJECT = Path(__file__).resolve().parents[1]
DATA = (
    Path("/tmp") / os.environ["USER"]
    / "glacial_lakes/nepal_glo/nepal_30_final_dataset"
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def scores(tp, fp, fn, tn):
    eps = 1e-8
    return {
        "dice": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        "iou": float((tp + eps) / (tp + fp + fn + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "fpr": float((fp + eps) / (fp + tn + eps)),
    }


def get_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported checkpoint format")

    for key in (
        "state",
        "model_state_dict",
        "state_dict",
        "model_state",
        "model",
    ):
        state = checkpoint.get(key)
        if isinstance(state, dict) and state:
            return state

    if checkpoint and all(
        torch.is_tensor(v) for v in checkpoint.values()
    ):
        return checkpoint

    raise RuntimeError(
        "Cannot locate model weights. "
        f"Checkpoint keys: {list(checkpoint.keys())}"
    )


def make_panel(rgb, gt, pred, out):
    rgb = rgb.convert("RGB")
    gt_rgb = Image.fromarray(gt.astype(np.uint8) * 255).convert("RGB")
    pr_rgb = Image.fromarray(pred.astype(np.uint8) * 255).convert("RGB")

    overlay = np.asarray(rgb).copy()
    gt_only = (gt == 1) & (pred == 0)
    fp_only = (gt == 0) & (pred == 1)
    correct = (gt == 1) & (pred == 1)

    overlay[correct] = [0, 210, 80]
    overlay[gt_only] = [255, 200, 0]
    overlay[fp_only] = [255, 40, 50]

    images = [
        rgb,
        gt_rgb,
        pr_rgb,
        Image.fromarray(overlay),
    ]

    canvas = Image.new("RGB", (rgb.width * 4, rgb.height + 30), "white")
    draw = ImageDraw.Draw(canvas)

    for i, (label, im) in enumerate(zip(
        ["RGB", "Ground Truth", "Prediction", "Error Overlay"],
        images,
    )):
        draw.text((i * rgb.width + 6, 8), label, fill="black")
        canvas.paste(im, (i * rgb.width, 30))

    canvas.save(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True,
        choices=["unet", "deeplab", "segformer"]
    )
    parser.add_argument(
        "--seed", required=True, type=int,
        choices=[42, 142, 242]
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    run_name = f"{args.model}_seed{args.seed}"
    ckpt_path = PROJECT / "results" / run_name / "best.pt"

    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    metadata_path = DATA / "metadata.csv"
    with open(metadata_path, newline="") as f:
        records = list(csv.DictReader(f))

    if args.limit:
        records = records[:args.limit]

    model_names = {
        "unet": "unet",
        "deeplab": "deeplabv3plus",
        "segformer": "segformer_b0",
    }

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        ckpt_path, map_location="cpu"
    )

    image_size = int(checkpoint.get(
        "size",
        352 if args.model == "unet" else 256
    ))

    model = build_model(model_names[args.model])
    model.load_state_dict(get_state_dict(checkpoint), strict=True)
    model = model.to(device).eval()

    print("Run:", run_name, flush=True)
    print("Device:", device, flush=True)
    print("Checkpoint:", ckpt_path, flush=True)
    print("Input size:", image_size, flush=True)
    print("Images:", len(records), flush=True)

    suffix = "smoke" if args.limit else "full30"
    out = PROJECT / "results" / "nepal30_eval" / suffix / run_name

    pred_dir = out / "predictions"
    panel_dir = out / "panels"
    pred_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for idx, rec in enumerate(records, 1):
        image_path = DATA / rec["IMAGE_PNG"]
        mask_path = DATA / rec["MASK_PNG"]

        rgb = Image.open(image_path).convert("RGB")
        gt = (
            np.asarray(
                Image.open(mask_path).convert("L")
            ) > 127
        ).astype(np.uint8)

        original_w, original_h = rgb.size

        resized = rgb.resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR
        )

        arr = np.asarray(resized).astype(np.float32) / 255.0
        arr = (arr - MEAN) / STD

        x = torch.from_numpy(
            arr.transpose(2, 0, 1).copy()
        ).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(x)

            if hasattr(logits, "logits"):
                logits = logits.logits

            if logits.ndim != 4 or logits.shape[1] != 1:
                raise RuntimeError(
                    f"Unexpected logits shape: {logits.shape}"
                )

            logits = F.interpolate(
                logits,
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False
            )

            pred = (
                torch.sigmoid(logits)[0, 0].cpu().numpy()
                >= 0.5
            ).astype(np.uint8)

        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        tn = int(((pred == 0) & (gt == 0)).sum())

        image_scores = scores(tp, fp, fn, tn)

        stem = Path(rec["IMAGE_PNG"]).stem

        Image.fromarray(pred * 255).save(
            pred_dir / f"{stem}_pred.png"
        )

        make_panel(
            rgb, gt, pred,
            panel_dir / f"{stem}_comparison.png"
        )

        rows.append({
            "model": args.model,
            "seed": args.seed,
            "BASIN": rec["BASIN"],
            "GLO_ID": rec["GLO_ID"],
            "DATE": rec["DATE"],
            "STAC_ID": rec["STAC_ID"],
            "VALID_RGB_FRACTION": rec["VALID_RGB_FRACTION"],
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            **image_scores,
        })

        print(
            f"[{idx}/{len(records)}] "
            f"{rec['BASIN']} "
            f"Dice={image_scores['dice']:.4f} "
            f"IoU={image_scores['iou']:.4f}",
            flush=True
        )

    with open(out / "image_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def summarize(group):
        totals = {
            k: sum(int(r[k]) for r in group)
            for k in ("TP", "FP", "FN", "TN")
        }
        return {
            "n_images": len(group),
            "pooled": scores(**{
                k.lower(): v for k, v in totals.items()
            }),
            "macro": {
                metric: float(np.mean([
                    float(r[metric]) for r in group
                ]))
                for metric in (
                    "dice", "iou", "precision", "recall", "fpr"
                )
            },
            "confusion_pixels": totals,
        }

    summary = {
        "model": args.model,
        "seed": args.seed,
        "threshold": 0.5,
        "input_size": image_size,
        "note": "Provisional dataset QA; not final paper metrics",
        "overall": summarize(rows),
        "by_basin": {
            basin: summarize([
                r for r in rows if r["BASIN"] == basin
            ])
            for basin in ("Gandaki", "Karnali", "Koshi")
        }
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(summary["overall"], indent=2))
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
