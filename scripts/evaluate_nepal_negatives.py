import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.models import build_model

ROOT = Path("/tmp") / __import__("os").environ["USER"] / "glacial_lakes/nepal_glo"
DATASETS = {
    "nepal_negative_candidates": ROOT / "nepal_negative_candidates",
    "koshi_negative_expansion": ROOT / "koshi_negative_expansion",
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["unet", "deeplab", "segformer"], required=True)
parser.add_argument("--seed", choices=[42, 142, 242], type=int, required=True)
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()

run = f"{args.model}_seed{args.seed}"
ckpt_path = PROJECT / "results" / run / "best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

state = ckpt["state"]
size = int(ckpt["size"])

names = {
    "unet": "unet",
    "deeplab": "deeplabv3plus",
    "segformer": "segformer_b0",
}

model = build_model(names[args.model])
model.load_state_dict(state, strict=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

df = pd.read_csv(PROJECT / "results/nepal_negative_provisional19.csv")
assert len(df) == 19
assert df["NEG_ID"].is_unique
assert (df["QA_STATUS"] == "PROVISIONAL_PASS").all()

if args.limit:
    df = df.head(args.limit)

output_group = "smoke" if args.limit else "full19"
out = PROJECT / "results/nepal_negative_eval" / output_group / run
(out / "predictions").mkdir(parents=True, exist_ok=True)

results = []

print("RUN:", run, flush=True)
print("DEVICE:", device, flush=True)
print("CHECKPOINT:", ckpt_path, flush=True)
print("INPUT SIZE:", size, flush=True)
print("N IMAGES:", len(df), flush=True)

for i, (_, rec) in enumerate(df.iterrows(), 1):
    source = str(rec["DATASET_SOURCE"])
    img_path = DATASETS[source] / str(rec["IMAGE_PNG"])

    rgb = Image.open(img_path).convert("RGB")
    w, h = rgb.size

    resized = rgb.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    x = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)
        if hasattr(logits, "logits"):
            logits = logits.logits
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
        logits = F.interpolate(
            logits, size=(h, w),
            mode="bilinear", align_corners=False
        )
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = (probability >= 0.5).astype(np.uint8)

    components, n_components = label(
        pred, structure=np.ones((3, 3), dtype=np.uint8)
    )

    fp_pixels = int(pred.sum())
    pixel_area_km2 = (10.0 * 10.0) / 1_000_000
    area_km2 = h * w * pixel_area_km2

    Image.fromarray(pred * 255).save(
        out / "predictions" / f"{rec['NEG_ID']}_pred.png"
    )

    results.append({
        "model": args.model,
        "seed": args.seed,
        "BASIN": rec["BASIN"],
        "NEG_ID": rec["NEG_ID"],
        "STAC_ID": rec["STAC_ID"],
        "DATASET_SOURCE": source,
        "FP_PIXELS": fp_pixels,
        "FP_FRACTION": fp_pixels / (w * h),
        "FP_PIXELS_PER_KM2": fp_pixels / area_km2,
        "FP_COMPONENTS": int(n_components),
        "FALSE_ALARM": int(fp_pixels > 0),
    })

    print(
        f"[{i}/{len(df)}] {rec['NEG_ID']} "
        f"FP_pixels={fp_pixels} "
        f"components={n_components}",
        flush=True
    )

metrics = pd.DataFrame(results)
metrics.to_csv(out / "image_metrics.csv", index=False)

summary = {
    "model": args.model,
    "seed": args.seed,
    "n_images": len(metrics),
    "threshold": 0.5,
    "total_fp_pixels": int(metrics.FP_PIXELS.sum()),
    "mean_fp_pixels_per_image": float(metrics.FP_PIXELS.mean()),
    "mean_fp_pixels_per_km2": float(metrics.FP_PIXELS_PER_KM2.mean()),
    "mean_fp_components_per_image": float(metrics.FP_COMPONENTS.mean()),
    "false_alarm_image_rate": float(metrics.FALSE_ALARM.mean()),
    "by_basin": {},
    "status": "PROVISIONAL_INVENTORY_NEGATIVE",
}

for basin, subset in metrics.groupby("BASIN"):
    summary["by_basin"][basin] = {
        "n_images": len(subset),
        "mean_fp_pixels": float(subset.FP_PIXELS.mean()),
        "mean_fp_components": float(subset.FP_COMPONENTS.mean()),
        "false_alarm_image_rate": float(subset.FALSE_ALARM.mean()),
    }

(out / "summary.json").write_text(json.dumps(summary, indent=2))

print("SUMMARY:")
print(json.dumps(summary, indent=2))
print("SAVED:", out)
