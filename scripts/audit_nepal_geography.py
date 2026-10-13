import os
import csv
from pathlib import Path
from itertools import combinations

import pandas as pd
import rasterio
from shapely.geometry import box
from shapely.ops import transform
from pyproj import Transformer

PROJECT = Path.home() / "Himalayan_Glacial_Lake_Segmentation"
ROOT = Path("/tmp") / os.environ["USER"] / "glacial_lakes/nepal_glo"
DATA = ROOT / "nepal_30_final_dataset"
OUT = PROJECT / "results/nepal_geospatial_audit"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "metadata.csv")

assert len(df) == 30, f"Expected 30 patches, found {len(df)}"

records = []
footprints = {}

# Transform all footprints to a shared metric CRS.
TARGET_CRS = "EPSG:32645"

for _, rec in df.iterrows():
    path = DATA / rec["GEOTIFF"]

    with rasterio.open(path) as ds:
        geom = box(*ds.bounds)

        transformer = Transformer.from_crs(
            ds.crs, TARGET_CRS, always_xy=True
        )

        projected = transform(transformer.transform, geom)

        key = str(rec["GLO_ID"])
        footprints[key] = projected

        records.append({
            "GLO_ID": key,
            "BASIN": rec["BASIN"],
            "STAC_ID": rec["STAC_ID"],
            "DATE": rec["DATE"],
            "CRS": str(ds.crs),
            "WIDTH": ds.width,
            "HEIGHT": ds.height,
            "AREA_KM2": projected.area / 1e6,
            "VALID_RGB_FRACTION": rec["VALID_RGB_FRACTION"],
        })

pd.DataFrame(records).to_csv(
    OUT / "nepal30_footprints.csv",
    index=False
)

overlaps = []

for a, b in combinations(records, 2):
    ga = footprints[a["GLO_ID"]]
    gb = footprints[b["GLO_ID"]]

    intersection = ga.intersection(gb)

    if intersection.area > 0:
        overlaps.append({
            "GLO_ID_A": a["GLO_ID"],
            "GLO_ID_B": b["GLO_ID"],
            "BASIN_A": a["BASIN"],
            "BASIN_B": b["BASIN"],
            "OVERLAP_AREA_M2": intersection.area,
            "OVERLAP_PERCENT_SMALLER": (
                100 * intersection.area / min(ga.area, gb.area)
            ),
        })

pd.DataFrame(
    overlaps,
    columns=[
        "GLO_ID_A", "GLO_ID_B",
        "BASIN_A", "BASIN_B",
        "OVERLAP_AREA_M2",
        "OVERLAP_PERCENT_SMALLER",
    ]
).to_csv(
    OUT / "nepal30_overlapping_pairs.csv",
    index=False
)

print("=== NEPAL-30 GEOSPATIAL AUDIT ===")
print("Total patches:", len(df))
print("Unique source scenes:", df.STAC_ID.nunique())
print("Overlapping patch pairs:", len(overlaps))

print("\n=== SCENES BY BASIN ===")
print(df.groupby("BASIN").STAC_ID.nunique())

print("\n=== OVERLAPPING PAIRS ===")
for overlap in overlaps:
    print(overlap)

print("\nResults saved:", OUT)
