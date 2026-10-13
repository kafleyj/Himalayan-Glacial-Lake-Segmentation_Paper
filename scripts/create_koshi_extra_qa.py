import csv
import os
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT = Path.home() / "Himalayan_Glacial_Lake_Segmentation"

DATA = (
    Path("/tmp") / os.environ["USER"]
    / "glacial_lakes/nepal_glo/koshi_negative_expansion"
)

OUT = PROJECT / "results/nepal_qa/negative_candidates"
OUT.mkdir(parents=True, exist_ok=True)

with open(DATA / "metadata.csv", newline="") as f:
    records = list(csv.DictReader(f))

print("New Koshi candidates:", len(records))

CELL_W = 280
CELL_H = 310
COLS = 4
ROWS = (len(records) + COLS - 1) // COLS

canvas = Image.new(
    "RGB",
    (COLS * CELL_W, ROWS * CELL_H + 55),
    "white"
)

draw = ImageDraw.Draw(canvas)

draw.text(
    (15, 15),
    "Koshi - Additional inventory-negative candidates",
    fill="black"
)

for i, rec in enumerate(records):
    col = i % COLS
    row = i // COLS

    x = col * CELL_W + 12
    y = row * CELL_H + 55

    img = Image.open(DATA / rec["IMAGE_PNG"]).convert("RGB")
    img = img.resize((256, 256))

    canvas.paste(img, (x, y))

    draw.text(
        (x, y + 262),
        rec["NEG_ID"],
        fill="black"
    )

    draw.text(
        (x, y + 279),
        f"Coverage: {float(rec['VALID_RGB_FRACTION']):.3f}",
        fill="black"
    )

out = OUT / "Koshi_additional_negative_qa.png"
canvas.save(out)

print("Saved:", out)
