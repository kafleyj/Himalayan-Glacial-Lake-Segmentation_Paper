from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path("results/nepal_pilot")
models = ["unet", "deeplab", "segformer"]
seed = 42

image_files = sorted((root / "unet_seed42/comparisons").glob("*_comparison.png"))

if len(image_files) != 3:
    raise RuntimeError(f"Expected 3 basin comparisons, found {len(image_files)}")

labels = ["Original RGB", "Ground Truth", "U-Net", "DeepLabV3+", "SegFormer-B0"]
tile = 256
header = 40
row_height = 286

canvas = Image.new("RGB", (5 * tile, header + 3 * row_height), "white")
draw = ImageDraw.Draw(canvas)

for col, label in enumerate(labels):
    draw.text((col * tile + 10, 12), label, fill="black")

for row, comparison in enumerate(image_files):
    stem = comparison.name.replace("_comparison.png", "")
    basin = stem.split("_")[0]

    # Existing comparison: RGB | GT | U-Net prediction
    source = Image.open(comparison).convert("RGB")
    rgb = source.crop((0, 34, 256, 290))
    gt = source.crop((256, 34, 512, 290))

    pred_images = []
    for model in models:
        p = root / f"{model}_seed{seed}" / "predictions" / f"{stem}_pred.png"
        pred_images.append(Image.open(p).convert("RGB"))

    pictures = [rgb, gt] + pred_images
    y = header + row * row_height

    draw.text((10, y + 3), basin, fill="black")
    for col, pic in enumerate(pictures):
        canvas.paste(pic, (col * tile, y + 30))

out = root / "nepal_three_basin_comparison_seed42.png"
canvas.save(out)
print("Saved:", out)
