import csv
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.ndimage import label

PROJECT = Path.home() / "Himalayan_Glacial_Lake_Segmentation"
DATA = (
    Path("/tmp") / os.environ["USER"]
    / "glacial_lakes/nepal_glo/nepal_30_final_dataset"
)
ROOT = PROJECT / "results/nepal30_eval"
OUT = ROOT / "size_analysis"
OUT.mkdir(parents=True, exist_ok=True)

with open(DATA / "metadata.csv", newline="") as f:
    records = list(csv.DictReader(f))

assert len(records) == 30, f"Expected 30 patches, got {len(records)}"

models = ("unet", "deeplab", "segformer")
seeds = (42, 142, 242)

def category(area):
    if area < 100:
        return "Small"
    if area < 500:
        return "Medium"
    return "Large"

rows = []

for model in models:
    for seed in seeds:
        name = f"{model}_seed{seed}"
        predictions = ROOT / "full30" / name / "predictions"

        for rec in records:
            gt_path = DATA / rec["MASK_PNG"]
            stem = Path(rec["IMAGE_PNG"]).stem
            pred_path = predictions / f"{stem}_pred.png"

            if not pred_path.is_file():
                raise FileNotFoundError(pred_path)

            gt = np.asarray(Image.open(gt_path).convert("L")) > 127
            pred = np.asarray(Image.open(pred_path).convert("L")) > 127

            if gt.shape != pred.shape:
                raise ValueError(
                    f"Shape mismatch: {gt_path} {pred_path}"
                )

            components, count = label(
                gt,
                structure=np.ones((3, 3), dtype=np.uint8)
            )

            for component_id in range(1, count + 1):
                region = components == component_id
                area = int(region.sum())
                hits = int((region & pred).sum())
                recall = hits / area

                rows.append({
                    "model": model,
                    "seed": seed,
                    "BASIN": rec["BASIN"],
                    "GLO_ID": rec["GLO_ID"],
                    "component_id": component_id,
                    "size_group": category(area),
                    "area_px": area,
                    "matched_px": hits,
                    "component_recall": recall,
                    "any_overlap": int(hits > 0),
                    "recall_10pct": int(recall >= 0.10),
                })

def save_csv(path, data, columns):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)

save_csv(
    OUT / "component_results.csv",
    rows,
    list(rows[0].keys())
)

summary = []

for model in models:
    for seed in seeds:
        for group in ("Small", "Medium", "Large"):
            subset = [
                r for r in rows
                if r["model"] == model
                and r["seed"] == seed
                and r["size_group"] == group
            ]

            if not subset:
                continue

            summary.append({
                "model": model,
                "seed": seed,
                "size_group": group,
                "components": len(subset),
                "any_overlap_rate": float(np.mean(
                    [r["any_overlap"] for r in subset]
                )),
                "detected_10pct_rate": float(np.mean(
                    [r["recall_10pct"] for r in subset]
                )),
                "mean_component_recall": float(np.mean(
                    [r["component_recall"] for r in subset]
                )),
                "area_weighted_recall": (
                    sum(r["matched_px"] for r in subset)
                    / sum(r["area_px"] for r in subset)
                )
            })

save_csv(
    OUT / "summary_by_seed.csv",
    summary,
    list(summary[0].keys())
)

print("GT components per run:", len(rows) // 9)
print("Component evaluations:", len(rows))
print()
print(
    f"{'Model':12} {'Seed':>4} {'Size':8} "
    f"{'N':>4} {'Det@10%':>9} {'MeanRecall':>11}"
)
print("-" * 59)

for r in summary:
    print(
        f"{r['model']:12} {r['seed']:4} "
        f"{r['size_group']:8} {r['components']:4} "
        f"{r['detected_10pct_rate']:9.3f} "
        f"{r['mean_component_recall']:11.3f}"
    )

print("\nSaved results to:", OUT)
