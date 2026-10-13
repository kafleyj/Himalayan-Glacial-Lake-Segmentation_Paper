from pathlib import Path
import csv
import numpy as np
from PIL import Image
from scipy.ndimage import label

root = Path("/tmp") / __import__("os").environ["USER"] / \
       "glacial_lakes/nepal_glo/nepal_30_final_dataset"

with open(root / "metadata.csv", newline="") as f:
    records = list(csv.DictReader(f))

rows = []

for rec in records:
    mask = np.array(
        Image.open(root / rec["MASK_PNG"]).convert("L")
    ) > 127

    components, n = label(
        mask, structure=np.ones((3, 3), dtype=np.uint8)
    )

    for i in range(1, n + 1):
        area_px = int((components == i).sum())

        rows.append({
            "BASIN": rec["BASIN"],
            "GLO_ID": rec["GLO_ID"],
            "component_id": i,
            "area_pixels": area_px,
            "area_m2_approx": area_px * 100,
        })

out = Path(
    "results/nepal30_eval/lake_component_sizes.csv"
)

with open(out, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

areas = np.array([r["area_pixels"] for r in rows])

print("Total connected components:", len(rows))
print("Small < 100 pixels:", int((areas < 100).sum()))
print("Medium 100-499 pixels:",
      int(((areas >= 100) & (areas < 500)).sum()))
print("Large >= 500 pixels:", int((areas >= 500).sum()))
print("Minimum component area:", areas.min())
print("Median component area:", np.median(areas))
print("Maximum component area:", areas.max())
print("Saved:", out)
