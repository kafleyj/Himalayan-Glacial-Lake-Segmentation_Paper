import csv
import os
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT = Path.home() / "Himalayan_Glacial_Lake_Segmentation"

DATA = (
    Path("/tmp") / os.environ["USER"]
    / "glacial_lakes/nepal_glo/nepal_negative_candidates"
)

OUT = PROJECT / "results/nepal_qa/negative_candidates"
OUT.mkdir(parents=True, exist_ok=True)

with open(DATA / "metadata.csv", newline="") as f:
    records = list(csv.DictReader(f))

print("Total candidates:", len(records))

THUMB = 240
COLS = 5
CELL_W = 260
CELL_H = 305

for basin in ("Gandaki", "Karnali", "Koshi"):
    subset = [r for r in records if r["BASIN"] == basin]

    if not subset:
        print(f"{basin}: no candidates")
        continue

    nrows = (len(subset) + COLS - 1) // COLS

    canvas = Image.new(
        "RGB",
        (COLS * CELL_W, nrows * CELL_H + 55),
        "white"
    )
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (15, 15),
        f"{basin} - Inventory-negative candidate QA ({len(subset)} images)",
        fill="black"
    )

    for i, rec in enumerate(subset):
        col = i % COLS
        row = i // COLS

        x = col * CELL_W + 10
        y = row * CELL_H + 55

        path = DATA / rec["IMAGE_PNG"]
        img = Image.open(path).convert("RGB")
        img = img.resize((THUMB, THUMB))

        canvas.paste(img, (x, y))

        draw.text(
            (x, y + THUMB + 5),
            rec["NEG_ID"],
            fill="black"
        )

        draw.text(
            (x, y + THUMB + 20),
            f"Coverage: {float(rec['VALID_RGB_FRACTION']):.3f}",
            fill="black"
        )

        draw.text(
            (x, y + THUMB + 35),
            f"Scene: {rec['DATE']}",
            fill="black"
        )

    output = OUT / f"{basin}_negative_qa.png"
    canvas.save(output)

    print(
        f"{basin}: {len(subset)} images -> {output}",
        flush=True
    )

print("QA sheets completed.")
