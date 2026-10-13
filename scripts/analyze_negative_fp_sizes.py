from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label

ROOT = Path("results/nepal_negative_eval/full19")
OUT = Path("results/nepal_negative_eval")
metadata = pd.read_csv("results/nepal_negative_provisional19.csv")

thresholds = (1, 10, 25, 100)
rows = []

for model in ("unet", "deeplab", "segformer"):
    for seed in (42, 142, 242):
        run = ROOT / f"{model}_seed{seed}"

        for _, rec in metadata.iterrows():
            path = run / "predictions" / f"{rec['NEG_ID']}_pred.png"

            if not path.exists():
                raise FileNotFoundError(path)

            pred = np.asarray(Image.open(path).convert("L")) > 127
            cc, count = label(
                pred, structure=np.ones((3, 3), dtype=np.uint8)
            )

            sizes = np.bincount(cc.ravel())[1:]

            for threshold in thresholds:
                kept = sizes[sizes >= threshold]

                rows.append({
                    "model": model,
                    "seed": seed,
                    "BASIN": rec["BASIN"],
                    "NEG_ID": rec["NEG_ID"],
                    "min_component_pixels": threshold,
                    "n_components": len(kept),
                    "fp_pixels_retained": int(kept.sum()),
                    "false_alarm": int(len(kept) > 0),
                })

df = pd.DataFrame(rows)
df.to_csv(
    OUT / "fp_component_threshold_details.csv", index=False
)

by_seed = (
    df.groupby(["model", "seed", "min_component_pixels"])
    .agg(
        alarm_rate=("false_alarm", "mean"),
        components_per_image=("n_components", "mean"),
        fp_pixels_per_image=("fp_pixels_retained", "mean")
    )
    .reset_index()
)

by_seed.to_csv(
    OUT / "fp_component_threshold_by_seed.csv", index=False
)

summary = (
    by_seed.groupby(["model", "min_component_pixels"])
    .agg(
        alarm_mean=("alarm_rate", "mean"),
        alarm_sd=("alarm_rate", "std"),
        components_mean=("components_per_image", "mean"),
        pixels_mean=("fp_pixels_per_image", "mean")
    )
    .reset_index()
)

summary.to_csv(
    OUT / "fp_component_threshold_summary.csv", index=False
)

print("\n=== FALSE-ALARM RATE BY MINIMUM COMPONENT SIZE ===")
table = summary.pivot(
    index="model",
    columns="min_component_pixels",
    values="alarm_mean"
) * 100

print(table.round(1).to_string())
print("\nSaved all CSV files.")
