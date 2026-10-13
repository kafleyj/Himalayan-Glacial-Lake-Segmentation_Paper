import os
import csv
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

root = Path("/tmp") / os.environ["USER"] / "glacial_lakes/nepal_glo/nepal_30_dataset"
out = Path.home() / "Himalayan_Glacial_Lake_Segmentation/results/nepal_test3_gt_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)

with open(root / "metadata_test3.csv", newline="") as f:
    records = list(csv.DictReader(f))

tile = 256
margin = 12
header = 70
row_h = 292

canvas = Image.new(
    "RGB",
    (3 * tile, header + len(records) * row_h),
    "white"
)
draw = ImageDraw.Draw(canvas)

for col, title in enumerate(
    ["Sentinel-2 RGB", "Ground Truth Mask", "GT Overlay"]
):
    draw.text((col * tile + margin, 20), title, fill="black")

for row, rec in enumerate(records):
    rgb = Image.open(root / rec["IMAGE_PNG"]).convert("RGB")
    mask = Image.open(root / rec["MASK_PNG"]).convert("L")
    overlay = Image.open(root / rec["OVERLAY_PNG"]).convert("RGB")

    # Make extremely small GT objects visible for visual inspection.
    mask_binary = np.asarray(mask) > 0
    mask_display = Image.fromarray(
        (mask_binary.astype(np.uint8) * 255), mode="L"
    ).convert("RGB")

    y = header + row * row_h
    title = (
        f"{rec['BASIN']} | GT pixels: {rec['LAKE_PIXELS']} "
        f"| {rec['DATE']}"
    )
    draw.text((margin, y), title, fill="black")

    for col, im in enumerate([rgb, mask_display, overlay]):
        canvas.paste(im, (col * tile, y + 25))

canvas.save(out)
print("Saved:", out)
