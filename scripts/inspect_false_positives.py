from pathlib import Path
from PIL import Image, ImageDraw
import rasterio
from rasterio.windows import Window

root = Path("/tmp") / "jeevan.kaphle@usd.local" / "glacial_lakes"
path = root / "extracted/transferability/Transferability validation/images/AS.tif"

coords = [
    (5888, 4096),
    (4608, 2816),
    (5376, 4352),
    (5632, 3840),
    (5120, 3328),
    (5888, 3840),
]

output = Path("results/transferability/unet_seed42/AS/samples")
output.mkdir(parents=True, exist_ok=True)

canvas = Image.new("RGB", (768, 560), "white")
draw = ImageDraw.Draw(canvas)

with rasterio.open(path) as ds:
    for i, (x, y) in enumerate(coords):
        rgb = ds.read(
            window=Window(x, y, 256, 256)
        ).transpose(1, 2, 0)

        image = Image.fromarray(rgb)

        cx = (i % 3) * 256
        cy = (i // 3) * 280

        canvas.paste(image, (cx, cy + 20))
        draw.text(
            (cx + 5, cy + 3),
            f"x={x}, y={y}",
            fill="black"
        )

out = output / "AS_high_false_positive_RGB.png"
canvas.save(out)
print("Saved:", out)
