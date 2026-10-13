import os
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pystac_client import Client
import planetary_computer

project = Path.cwd()
root = Path("/tmp") / os.environ["USER"] / "glacial_lakes/nepal_glo"
output = project / "results/nepal_qa/alternative_scenes"
output.mkdir(parents=True, exist_ok=True)

bad_ids = [
    "GLO_83.64315_29.02262",
    "GLO_83.65418_29.08032",
    "GLO_83.50758_28.81072",
    "GLO_82.03939_30.31141",
    "GLO_87.81105_27.91592",
]

df = pd.read_csv(
    project / "results/nepal_90_scene_candidates.csv"
)
df = df[df["GLO_ID"].isin(bad_ids)]

# Import existing RGB preprocessing without running its main().
os.chdir(root)
spec = importlib.util.spec_from_file_location(
    "pilot_builder",
    root / "build_nepal_pilot_dataset.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
os.chdir(project)

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

records = []

for glo_id, group in df.groupby("GLO_ID", sort=True):
    print("\n=== ", glo_id, " ===", flush=True)

    previews = []

    for _, row in group.iterrows():
        try:
            items = list(
                catalog.search(
                    collections=["sentinel-2-l2a"],
                    ids=[row["STAC_ID"]]
                ).items()
            )
            if not items:
                raise RuntimeError("Scene not found")

            item = planetary_computer.sign(items[0])

            rgb, _, _, _ = builder.read_rgb_patch(
                item,
                float(row["LONGITUDE"]),
                float(row["LATITUDE"])
            )

            valid = np.all(rgb > 0, axis=0)
            coverage = float(valid.mean())

            rgb8 = builder.to_uint8(rgb)

            preview = Image.fromarray(rgb8)
            previews.append((
                str(row["DATE"]),
                coverage,
                preview
            ))

            records.append({
                "GLO_ID": glo_id,
                "BASIN": row["BASIN"],
                "STAC_ID": row["STAC_ID"],
                "DATE": row["DATE"],
                "CLOUD_COVER": row["CLOUD_COVER"],
                "RGB_COVERAGE": coverage,
                "STATUS": "CHECKED",
                "ERROR": ""
            })

            print(
                f"{row['DATE']}: "
                f"coverage={coverage:.3f}",
                flush=True
            )

        except Exception as e:
            records.append({
                "GLO_ID": glo_id,
                "BASIN": row["BASIN"],
                "STAC_ID": row["STAC_ID"],
                "DATE": row["DATE"],
                "CLOUD_COVER": row["CLOUD_COVER"],
                "RGB_COVERAGE": None,
                "STATUS": "ERROR",
                "ERROR": repr(e)
            })
            print("ERROR:", repr(e), flush=True)

    if previews:
        canvas = Image.new(
            "RGB", (256 * len(previews), 295), "white"
        )
        draw = ImageDraw.Draw(canvas)

        for i, (date, coverage, preview) in enumerate(previews):
            canvas.paste(preview, (i * 256, 35))
            draw.text(
                (i * 256 + 5, 10),
                f"{date} | valid={coverage:.3f}",
                fill="black"
            )

        canvas.save(output / f"{glo_id}_alternatives.png")

report = pd.DataFrame(records)
report.to_csv(output / "coverage_report.csv", index=False)

print("\n=== BEST COVERAGE PER LAKE ===")
valid = report.dropna(subset=["RGB_COVERAGE"])

if not valid.empty:
    best = (
        valid.sort_values(
            ["RGB_COVERAGE", "CLOUD_COVER"],
            ascending=[False, True]
        )
        .drop_duplicates("GLO_ID")
    )
    print(
        best[["GLO_ID", "DATE", "RGB_COVERAGE"]]
        .to_string(index=False)
    )

print("\nSaved:", output)
