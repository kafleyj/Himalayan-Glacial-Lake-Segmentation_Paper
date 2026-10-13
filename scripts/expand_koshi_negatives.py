import os
import csv
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import planetary_computer
from PIL import Image
from pyproj import Transformer
from pystac_client import Client
from shapely.geometry import box
from shapely.ops import transform as shp_transform, unary_union

PROJECT = Path.home() / "Himalayan_Glacial_Lake_Segmentation"
ROOT = Path("/tmp") / os.environ["USER"] / "glacial_lakes/nepal_glo"
OUT = ROOT / "koshi_negative_expansion"
OUT.joinpath("images").mkdir(parents=True, exist_ok=True)

SELECTION = PROJECT / "results/nepal_30_final_scene_selection.csv"
EXISTING = PROJECT / "results/nepal_negative_candidates.csv"

TARGET = 8
PATCH = 256
MIN_COVERAGE = 0.98
BUFFER_M = 100
MIN_CENTER_DISTANCE_M = 3000

os.chdir(ROOT)

spec = importlib.util.spec_from_file_location(
    "pilot_builder", ROOT / "build_nepal_pilot_dataset.py"
)
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)

polygons = builder.load_nepal_polygons()
selection = pd.read_csv(SELECTION)
selection = selection[selection["BASIN"] == "Koshi"]

old = pd.read_csv(EXISTING)
existing_centers = list(zip(
    old["LONGITUDE"].astype(float),
    old["LATITUDE"].astype(float)
))

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

scene_cache = {}
inventory_cache = {}
records = []

# Multiple directions and increasing distances.
offsets = []
for distance in (400, 600, 800, 1000, 1200, 1500):
    offsets.extend([
        (distance, 0), (-distance, 0),
        (0, distance), (0, -distance),
        (distance, distance), (-distance, distance),
        (distance, -distance), (-distance, -distance),
    ])


def get_scene(scene_id):
    if scene_id not in scene_cache:
        found = list(catalog.search(
            collections=["sentinel-2-l2a"],
            ids=[scene_id]
        ).items())

        if not found:
            raise RuntimeError(f"Scene not found: {scene_id}")

        scene_cache[scene_id] = planetary_computer.sign(found[0])

    return scene_cache[scene_id]


def get_inventory(crs):
    key = str(crs)

    if key not in inventory_cache:
        convert = Transformer.from_crs(
            "ESRI:102025", crs, always_xy=True
        ).transform

        shapes = [
            shp_transform(convert, p["geometry"])
            for p in polygons
            if p["geometry"] is not None
            and not p["geometry"].is_empty
        ]

        inventory_cache[key] = unary_union(shapes)

    return inventory_cache[key]


def sufficiently_separated(lon, lat, crs):
    convert = Transformer.from_crs(
        "EPSG:4326", crs, always_xy=True
    )
    x, y = convert.transform(lon, lat)

    for old_lon, old_lat in existing_centers:
        ox, oy = convert.transform(old_lon, old_lat)
        distance = ((x - ox)**2 + (y - oy)**2)**0.5
        if distance < MIN_CENTER_DISTANCE_M:
            return False

    return True


for _, row in selection.iterrows():
    if len(records) >= TARGET:
        break

    scene_id = str(row["STAC_ID"])
    item = get_scene(scene_id)

    with rasterio.open(item.assets["B04"].href) as ds:
        to_crs = Transformer.from_crs(
            "EPSG:4326", ds.crs, always_xy=True
        )
        from_crs = Transformer.from_crs(
            ds.crs, "EPSG:4326", always_xy=True
        )

        x, y = to_crs.transform(
            float(row["LONGITUDE"]),
            float(row["LATITUDE"])
        )
        base_row, base_col = ds.index(x, y)

        for dc, dr in offsets:
            if len(records) >= TARGET:
                break

            r = base_row + dr
            c = base_col + dc

            # Avoid obvious off-image candidates.
            if not (
                128 <= r < ds.height - 128
                and 128 <= c < ds.width - 128
            ):
                continue

            cx, cy = ds.xy(r, c)
            lon, lat = from_crs.transform(cx, cy)

            if not sufficiently_separated(lon, lat, ds.crs):
                continue

            try:
                rgb, transform, bounds, crs = (
                    builder.read_rgb_patch(item, lon, lat)
                )

                footprint = box(*bounds)

                # Entire footprint plus 100m buffer
                # must be free from inventory polygons.
                if get_inventory(crs).intersects(
                    footprint.buffer(BUFFER_M)
                ):
                    continue

                coverage = float(
                    np.any(rgb != 0, axis=0).mean()
                )
                if coverage < MIN_COVERAGE:
                    continue

                rgb8 = builder.to_uint8(rgb)

                if rgb8.shape != (PATCH, PATCH, 3):
                    continue

                neg_id = f"Koshi_EXTRA_{len(records)+1:02d}"
                filename = neg_id + ".png"

                Image.fromarray(rgb8).save(
                    OUT / "images" / filename
                )

                records.append({
                    "BASIN": "Koshi",
                    "NEG_ID": neg_id,
                    "LONGITUDE": lon,
                    "LATITUDE": lat,
                    "STAC_ID": scene_id,
                    "DATE": str(row["DATE"]),
                    "SOURCE_GLO_ID": str(row["GLO_ID"]),
                    "VALID_RGB_FRACTION": coverage,
                    "INVENTORY_EXCLUSION_BUFFER_M": BUFFER_M,
                    "IMAGE_PNG": "images/" + filename,
                    "QA_STATUS": "PENDING_VISUAL_REVIEW",
                })

                existing_centers.append((lon, lat))

                print(
                    f"{neg_id}: coverage={coverage:.3f}, "
                    f"source={row['GLO_ID']}",
                    flush=True
                )

            except Exception as exc:
                print(
                    f"Candidate skipped: {exc}",
                    flush=True
                )

if records:
    with open(OUT / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    pd.DataFrame(records).to_csv(
        PROJECT / "results/koshi_negative_expansion.csv",
        index=False
    )

print(f"NEW KOSHI CANDIDATES: {len(records)}")
print("Output:", OUT)
print("All candidates require manual QA.")
