import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from src.models import build_model
from src.metrics import scores

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["unet", "deeplab", "segformer"])
    parser.add_argument("--seed", type=int, default=42, choices=[42,142,242])
    args = parser.parse_args()

    root = Path.home() / "Himalayan_Glacial_Lake_Segmentation"
    data = Path("/tmp") / os.environ["USER"] / "glacial_lakes/nepal_glo/pilot_dataset"

    run_name = f"{args.model}_seed{args.seed}"
    checkpoint = root / "results" / run_name / "best.pt"

    out = root / "results/nepal_pilot" / run_name
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "comparisons").mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    # CPU avoids competing with GPU training jobs.
    torch.set_num_threads(2)
    ck = torch.load(checkpoint, map_location="cpu",
                    weights_only=False)

    model = build_model(ck["architecture"])
    model.load_state_dict(ck["state"], strict=True)
    model.eval()

    size = int(ck["size"])
    rows = []
    totals = dict(tp=0, tn=0, fp=0, fn=0)

    image_paths = sorted((data / "images").glob("*.png"))
    if not image_paths:
        raise RuntimeError("No Nepal pilot images found")

    for image_path in image_paths:
        stem = image_path.stem
        mask_path = data / "masks" / f"{stem}_mask.png"

        original = Image.open(image_path).convert("RGB")
        ground_truth = np.asarray(
            Image.open(mask_path).convert("L")
        ) > 0

        # Exactly match training image preprocessing.
        resized = original.resize(
            (size, size), Image.Resampling.BILINEAR
        )
        x = np.asarray(resized, dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        x = torch.from_numpy(
            x.transpose(2, 0, 1).copy()
        ).unsqueeze(0)

        with torch.inference_mode():
            logits = model(x)

            # Restore predictions to original 256x256
            # Nepal patch grid before scoring.
            if logits.shape[-2:] != ground_truth.shape:
                logits = F.interpolate(
                    logits,
                    size=ground_truth.shape,
                    mode="bilinear",
                    align_corners=False,
                )

            prob = torch.sigmoid(logits)[0, 0].numpy()

        pred = prob >= 0.5
        metric = scores(
            ground_truth.astype(np.float32),
            prob,
            threshold=0.5
        )

        basin = stem.split("_")[0]
        rows.append({
            "basin": basin,
            "image": stem,
            **metric
        })

        for key in totals:
            totals[key] += metric[key]

        Image.fromarray(
            (pred.astype(np.uint8) * 255)
        ).save(out / "predictions" / f"{stem}_pred.png")

        Image.fromarray(
            np.uint8(np.clip(prob * 255, 0, 255))
        ).save(out / "predictions" / f"{stem}_prob.png")

        # Comparison: RGB | GLO mask | model prediction
        panel = Image.new("RGB", (768, 290), "white")
        draw = ImageDraw.Draw(panel)

        images = [
            original,
            Image.fromarray(
                ground_truth.astype(np.uint8) * 255
            ).convert("RGB"),
            Image.fromarray(
                pred.astype(np.uint8) * 255
            ).convert("RGB"),
        ]

        for i, image in enumerate(images):
            panel.paste(image, (256 * i, 34))

        draw.text(
            (10, 10),
            f"{basin} | RGB / GLO mask / {args.model}",
            fill="black"
        )

        panel.save(
            out / "comparisons" / f"{stem}_comparison.png"
        )

        print(
            f"{basin}: Dice={metric['dice']:.4f}, "
            f"IoU={metric['iou']:.4f}",
            flush=True
        )

    tp, tn, fp, fn = (
        totals[key] for key in ("tp", "tn", "fp", "fn")
    )

    summary = {
        "model": ck["architecture"],
        "seed": ck["seed"],
        "n_images": len(rows),
        "dice": 2 * tp / (2 * tp + fp + fn + 1e-8),
        "iou": tp / (tp + fp + fn + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
        "specificity": tn / (tn + fp + 1e-8),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + 1e-8),
        **totals,
    }

    with open(out / "image_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    (out / "metrics.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\n=== NEPAL PILOT SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
