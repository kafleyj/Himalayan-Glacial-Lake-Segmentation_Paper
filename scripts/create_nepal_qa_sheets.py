import os
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path("/tmp") / os.environ["USER"] / "glacial_lakes/nepal_glo/nepal_30_dataset"
output = Path.home() / "Himalayan_Glacial_Lake_Segmentation/results/nepal_qa"
output.mkdir(parents=True, exist_ok=True)

with open(root / "metadata.csv", newline="") as f:
    records = list(csv.DictReader(f))

tile = 256
label_h = 48
head_h = 55
row_h = tile + label_h

for basin in ("Gandaki", "Karnali", "Koshi"):
    subset = [r for r in records if r["BASIN"] == basin]

    canvas = Image.new(
        "RGB",
        (tile * 3, head_h + len(subset) * row_h),
        "white"
    )
    draw = ImageDraw.Draw(canvas)

    for c, heading in enumerate(("RGB", "Ground Truth", "GT Overlay")):
        draw.text((c * tile + 12, 14), heading, fill="black")

    for i, rec in enumerate(subset):
        y = head_h + i * row_h

        label = (
            f"{i+1:02d} | {rec['GLO_ID']} "
            f"| pixels={rec['LAKE_PIXELS']}"
        )
        draw.text((10, y + 3), label, fill="black")

        for c, field in enumerate(
            ("IMAGE_PNG", "MASK_PNG", "OVERLAY_PNG")
        ):
            im = Image.open(root / rec[field]).convert("RGB")
            if im.size != (tile, tile):
                im = im.resize((tile, tile))

            canvas.paste(im, (c * tile, y + label_h))

    filename = output / f"{basin}_qa_sheet.png"
    canvas.save(filename)
    print("Saved:", filename, "samples:", len(subset))

print("\nQA sheets complete.")
