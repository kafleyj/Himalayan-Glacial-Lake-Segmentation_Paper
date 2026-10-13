import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT = Path.home() / "Himalayan_Glacial_Lake_Segmentation"
ROOT = Path("/tmp") / os.environ["USER"] / "glacial_lakes/nepal_glo"
OUT = PROJECT / "results/nepal_qa/negative_full_review"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    PROJECT / "results/nepal_negative_provisional21.csv"
)

assert len(df) == 21, f"Expected 21 samples, found {len(df)}"
assert df["NEG_ID"].is_unique, "Duplicate negative IDs"

roots = {
    "nepal_negative_candidates": ROOT / "nepal_negative_candidates",
    "koshi_negative_expansion": ROOT / "koshi_negative_expansion",
}

records = []

for basin in ("Gandaki", "Karnali", "Koshi"):
    subset = df[df["BASIN"] == basin]

    rows = len(subset)
    canvas = Image.new("RGB", (1080, rows * 550 + 50), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (15, 15),
        f"{basin} - Full-resolution negative candidate review",
        fill="black"
    )

    for j, (_, rec) in enumerate(subset.iterrows()):
        data_root = roots[rec["DATASET_SOURCE"]]
        image_path = data_root / rec["IMAGE_PNG"]

        img = Image.open(image_path).convert("RGB")

        assert img.size == (256, 256), (
            rec["NEG_ID"], img.size
        )

        rgb = np.asarray(img)
        rgb_coverage = float(np.any(rgb != 0, axis=2).mean())

        enlarged = img.resize(
            (512, 512),
            Image.Resampling.NEAREST
        )

        y = j * 550 + 45

        canvas.paste(enlarged, (10, y))
        draw.text(
            (540, y + 20),
            f"ID: {rec['NEG_ID']}",
            fill="black"
        )
        draw.text(
            (540, y + 50),
            f"Scene: {rec['DATE']}",
            fill="black"
        )
        draw.text(
            (540, y + 80),
            f"Coverage recorded: "
            f"{float(rec['VALID_RGB_FRACTION']):.3f}",
            fill="black"
        )
        draw.text(
            (540, y + 110),
            f"PNG nonblack fraction: {rgb_coverage:.3f}",
            fill="black"
        )
        draw.text(
            (540, y + 140),
            "QA: PROVISIONAL - inspect water/ice/shadow",
            fill="black"
        )

        records.append({
            **rec.to_dict(),
            "IMAGE_EXISTS": True,
            "IMAGE_WIDTH": img.width,
            "IMAGE_HEIGHT": img.height,
            "PNG_NONBLACK_FRACTION": rgb_coverage,
            "FULL_RES_QA_STATUS": "PENDING",
        })

    output = OUT / f"{basin}_negative_full_review.png"
    canvas.save(output)
    print("Saved:", output, flush=True)

pd.DataFrame(records).to_csv(
    OUT / "verification_metadata.csv",
    index=False
)

print("Samples checked:", len(records))
print("All images 256x256.")
print("Full-resolution visual review pending.")
