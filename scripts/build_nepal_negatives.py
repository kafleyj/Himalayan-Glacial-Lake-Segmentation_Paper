import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import planetary_computer
import rasterio
from PIL import Image
from pyproj import Transformer
from pystac_client import Client
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box
from shapely.ops import transform as shp_transform, unary_union

PROJECT = Path.home() / "Himalayan_Glacial_Lake_Segmentation"
ROOT = Path("/tmp") / os.environ["USER"] / "glacial_lakes/nepal_glo"
OUT = ROOT / "nepal_negative_candidates"
SELECTION = PROJECT / "results/nepal_30_final_scene_selection.csv"

TARGET_PER_BASIN = 10
PATCH_SIZE = 256
BUFFER_METERS = 100
MIN_COVERAGE = 0.98

(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "masks").mkdir(parents=True, exist_ok=True)

os.chdir(ROOT)
spec = importlib.util.spec_from_file_location(
    "pilot_builder", ROOT / "build_nepal_pilot_dataset.py"
)
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)

polygons = builder.load_nepal_polygons()
selection = pd.read_csv(SELECTION)

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# Pixel offsets from lake-centered locations.
# At nominal 10 m resolution, 400 px is about 4 km.
offsets = [
    (400, 0), (-400, 0), (0, 400), (0, -400),
    (500, 500), (-500, 500), (500, -500), (-500, -500),
    (700, 0), (-700, 0), (0, 700), (0, -700),
    (900, 0), (-900, 0), (0, 900), (0, -900),
]

scene_cache = {}
inventory_cache = {}
selected_footprints = []
records = []


def get_item(scene_id):
    if scene_id not in scene_cache:
        items = list(catalog.search(
            collections=["sentinel-2-l2a"],
            ids=[scene_id]
        ).items())
        if not items:
            raise RuntimeError(f"Missing STAC scene: {scene_id}")

        scene_cache[scene_id] = planetary_computer.sign(items[0])

    return scene_cache[scene_id]


def inventory_union(target_crs):
    key = str(target_crs)

    if key not in inventory_cache:
        to_target = Transformer.from_crs(
            "ESRI:102025", target_crs, always_xy=True
        ).transform

        geometries = [
            shp_transform(to_target, p["geometry"])
            for p in polygons
            if p["geometry"] is not None
            and not p["geometry"].is_empty
        ]

        inventory_cache[key] = unary_union(geometries)

    return inventory_cache[key]


def candidate_window(ds, lon, lat, dc, dr):
    transformer = Transformer.from_crs(
        "EPSG:4326", ds.crs, always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    base_row, base_col = ds.index(x, y)

    row = base_row + dr
    col = base_col + dc

    window = Window(
        col - PATCH_SIZE // 2,
        row - PATCH_SIZE // 2,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    bounds = window_bounds(window, ds.transform)
    cx, cy = ds.xy(row, col)

    to_geographic = Transformer.from_crs(
        ds.crs, "EPSG:4326", always_xy=True
    )
    new_lon, new_lat = to_geographic.transform(cx, cy)

    return bounds, new_lon, new_lat


def is_spatially_clear(footprint, crs, chosen):
    inventory = inventory_union(crs)

    expanded = footprint.buffer(BUFFER_METERS)

    if inventory.intersects(expanded):
        return False

    for previous_crs, previous_footprint in chosen:
        if str(previous_crs) != str(crs):
            # Conservative: do not compare geometries in different CRS.
            # Existing inventory still checks the candidate.
            continue

        if previous_footprint.buffer(BUFFER_METERS).intersects(
            footprint
        ):
            return False

    return True


for basin in ("Gandaki", "Karnali", "Koshi"):
    sources = selection[selection["BASIN"] == basin]
    basin_count = 0

    for _, source in sources.iterrows():
        if basin_count >= TARGET_PER_BASIN:
            break

        scene_id = str(source["STAC_ID"])
        item = get_item(scene_id)

        with rasterio.open(item.assets["B04"].href) as ds:
            for dc, dr in offsets:
                bounds, lon, lat = candidate_window(
                    ds,
                    float(source["LONGITUDE"]),
                    float(source["LATITUDE"]),
                    dc,
                    dr
                )

                footprint = box(*bounds)

                if not is_spatially_clear(
                    footprint, ds.crs, selected_footprints
                ):
                    continue

                try:
                    rgb, transform, actual_bounds, crs = (
                        builder.read_rgb_patch(item, lon, lat)
                    )

                    coverage = float(
                        np.any(rgb != 0, axis=0).mean()
                    )

                    if coverage < MIN_COVERAGE:
                        continue

                    # Full footprint + 100 m buffer has already
                    # passed the inventory-polygon exclusion test.
                    # An empty geometry list must not be rasterized.
                    actual_footprint = box(*actual_bounds)
                    if not is_spatially_clear(
                        actual_footprint, crs, selected_footprints
                    ):
                        continue

                    mask = np.zeros((256, 256), dtype=np.uint8)
                    n_poly = 0

                    rgb8 = builder.to_uint8(rgb)

                    if rgb8.shape != (256, 256, 3):
                        continue

                    name = f"{basin}_NEG_{basin_count + 1:02d}"
                    image_name = name + ".png"
                    mask_name = name + "_mask.png"

                    Image.fromarray(rgb8).save(
                        OUT / "images" / image_name
                    )
                    Image.fromarray(
                        np.zeros((256, 256), dtype=np.uint8)
                    ).save(OUT / "masks" / mask_name)

                    records.append({
                        "BASIN": basin,
                        "NEG_ID": name,
                        "LONGITUDE": lon,
                        "LATITUDE": lat,
                        "STAC_ID": scene_id,
                        "DATE": str(source["DATE"]),
                        "SOURCE_GLO_ID": str(source["GLO_ID"]),
                        "VALID_RGB_FRACTION": coverage,
                        "INVENTORY_POLYGON_COUNT": n_poly,
                        "INVENTORY_EXCLUSION_BUFFER_M": BUFFER_METERS,
                        "IMAGE_PNG": "images/" + image_name,
                        "MASK_PNG": "masks/" + mask_name,
                        "QA_STATUS": "PENDING_VISUAL_REVIEW",
                    })

                    selected_footprints.append((crs, footprint))
                    basin_count += 1

                    print(
                        f"{name}: coverage={coverage:.3f} "
                        f"source={source['GLO_ID']}",
                        flush=True
                    )
                    break

                except Exception as exc:
                    print(
                        f"Candidate error {basin}: {exc}",
                        flush=True
                    )

    print(
        f"{basin}: {basin_count}/{TARGET_PER_BASIN}",
        flush=True
    )

if records:
    columns = list(records[0].keys())
    with open(OUT / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    pd.DataFrame(records).to_csv(
        PROJECT / "results/nepal_negative_candidates.csv",
        index=False
    )

print(f"TOTAL CANDIDATES: {len(records)}", flush=True)
print("Dataset:", OUT, flush=True)
print(
    "NOTE: Inventory-negative candidates only. "
    "Visual verification is still required.",
    flush=True
)
