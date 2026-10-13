from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

project = Path.home() / "Himalayan_Glacial_Lake_Segmentation"
root = project / "results/nepal30_eval/full30/unet_seed42"

df = pd.read_csv(root / "image_metrics.csv")

cases = [
    ("Gandaki", "GLO_85.4078_28.08038", "Success"),
    ("Karnali", "GLO_83.17886_28.96383", "Success"),
    ("Koshi", "GLO_86.72477_27.99197", "Failure"),
    ("Karnali", "GLO_82.03939_30.31141", "Failure"),
    ("Gandaki", "GLO_84.13782_28.92426", "Failure"),
    ("Koshi", "GLO_86.6777_27.96456", "Success"),
]

panels = []

for basin, lake_id, category in cases:
    matching = df[
        (df["BASIN"] == basin) &
        (df["GLO_ID"] == lake_id)
    ]

    if len(matching) != 1:
        print("SKIPPING unmatched:", basin, lake_id)
        continue

    row = matching.iloc[0]
    matches = list((root / "panels").glob(f"*{lake_id}*comparison.png"))

    if not matches:
        print("Panel not found:", lake_id)
        continue

    img = Image.open(matches[0]).convert("RGB")
    panels.append((basin, lake_id, category, row["dice"], img))

if not panels:
    raise RuntimeError("No matching panels found")

width = max(p[4].width for p in panels)
height = max(p[4].height for p in panels) + 42

canvas = Image.new(
    "RGB",
    (width, height * len(panels)),
    "white"
)

draw = ImageDraw.Draw(canvas)

for i, (basin, lake_id, category, dice, img) in enumerate(panels):
    y = i * height

    draw.text(
        (10, y + 8),
        f"{category} | {basin} | {lake_id} | Dice={dice:.4f}",
        fill="black"
    )

    canvas.paste(img, (0, y + 42))

out = project / "results/nepal30_eval/nepal_failure_examples.png"
canvas.save(out)

print("Saved:", out)
print("Included panels:", len(panels))
