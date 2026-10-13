import os
import sys
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pystac_client import Client
import planetary_computer

project = Path.cwd()
root = Path("/tmp") / os.environ["USER"] / "glacial_lakes/nepal_glo"
selection = project / "results/nepal_30_repaired_selection.csv"

out = root / "nepal_30_repaired_dataset"
for sub in ["images", "masks", "overlays", "geotiff"]:
    (out / sub).mkdir(parents=True, exist_ok=True)

# Reuse the exact rasterization and RGB functions from the pilot.
os.chdir(root)
spec = importlib.util.spec_from_file_location(
    "pilot_builder", root / "build_nepal_pilot_dataset.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

polygons = builder.load_nepal_polygons()
df = pd.read_csv(selection)

limit = int(os.environ.get("NEPAL_LIMIT", "30"))
if limit == 3:
    df = df.groupby("BASIN", sort=True).head(1)
else:
    df = df.head(limit)

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

records = []
errors = []
cache = {}

for i, (_, row) in enumerate(df.iterrows(), 1):
    basin = str(row["BASIN"])
    glo_id = str(row["GLO_ID"])
    scene_id = str(row["STAC_ID"])
    lon = float(row["LONGITUDE"])
    lat = float(row["LATITUDE"])

    print(f"[{i}/{len(df)}] {basin}: {glo_id}", flush=True)

    try:
        if scene_id not in cache:
            items = list(catalog.search(
                collections=["sentinel-2-l2a"],
                ids=[scene_id]
            ).items())
            if not items:
                raise RuntimeError("STAC scene not found")
            cache[scene_id] = planetary_computer.sign(items[0])

        item = cache[scene_id]

        rgb, transform, bounds, crs = builder.read_rgb_patch(
            item, lon, lat
        )
        mask, n_polygons = builder.build_mask(
            polygons, bounds, crs, transform
        )
        rgb8 = builder.to_uint8(rgb)

        if rgb8.shape != (256, 256, 3):
            raise ValueError(f"Bad image shape {rgb8.shape}")
        if mask.shape != (256, 256):
            raise ValueError(f"Bad mask shape {mask.shape}")

        stem = f"{basin}_{glo_id}_{row['DATE']}"
        image_name = f"{stem}.png"
        mask_name = f"{stem}_mask.png"
        overlay_name = f"{stem}_overlay.png"
        tif_name = f"{stem}.tif"

        Image.fromarray(rgb8).save(out / "images" / image_name)
        Image.fromarray((mask > 0).astype(np.uint8) * 255).save(
            out / "masks" / mask_name
        )
        builder.save_overlay(
            out / "overlays" / overlay_name, rgb8, mask
        )
        builder.save_geotiff(
            out / "geotiff" / tif_name, rgb, transform, crs
        )

        lake_pixels = int((mask > 0).sum())
        valid_rgb_fraction = float(
            np.any(rgb != 0, axis=0).mean()
        )

        records.append({
            "BASIN": basin,
            "GLO_ID": glo_id,
            "STAC_ID": scene_id,
            "DATE": str(row["DATE"]),
            "CLOUD_COVER": row["CLOUD_COVER"],
            "size_group": row["size_group"],
            "N_POLYGONS_IN_PATCH": n_polygons,
            "LAKE_PIXELS": lake_pixels,
            "LAKE_FRACTION": lake_pixels / mask.size,
            "VALID_RGB_FRACTION": valid_rgb_fraction,
            "IMAGE_PNG": str(Path("images") / image_name),
            "MASK_PNG": str(Path("masks") / mask_name),
            "OVERLAY_PNG": str(Path("overlays") / overlay_name),
            "GEOTIFF": str(Path("geotiff") / tif_name),
            "QA_STATUS": "PENDING_VISUAL_REVIEW",
        })

        print(
            f"  Lake pixels={lake_pixels}, "
            f"polygons={n_polygons}, "
            f"valid RGB={valid_rgb_fraction:.3f}",
            flush=True
        )

    except Exception as exc:
        errors.append((glo_id, repr(exc)))
        print(f"  ERROR: {exc}", flush=True)

metadata = out / (
    "metadata_test3.csv" if limit == 3 else "metadata.csv"
)
if records:
    with open(metadata, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

print("\n=== BUILD SUMMARY ===")
print("Requested:", len(df))
print("Created:", len(records))
print("Errors:", len(errors))
print("Output:", out)
for glo_id, error in errors:
    print("FAILED:", glo_id, error)

if errors:
    sys.exit(1)
