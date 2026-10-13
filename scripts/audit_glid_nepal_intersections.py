import os
from pathlib import Path

import pandas as pd
import rasterio
import shapefile
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform

PROJECT = Path.home() / "Himalayan_Glacial_Lake_Segmentation"
ROOT = Path("/tmp") / os.environ["USER"] / "glacial_lakes"

SHAPEFILE = ROOT / "extracted/annotations/GLID_annotation.shp"
NEPAL = ROOT / "nepal_glo/nepal_30_final_dataset"

OUT = PROJECT / "results/nepal_geospatial_audit"
OUT.mkdir(parents=True, exist_ok=True)

reader = shapefile.Reader(str(SHAPEFILE))

glid_polygons = []
invalid_records = []

for i, record in enumerate(reader.iterShapeRecords()):
    try:
        geom = shape(record.shape.__geo_interface__)

        if geom.is_empty:
            raise ValueError("Empty geometry")

        if not geom.is_valid:
            geom = geom.buffer(0)

        if geom.is_empty or not geom.is_valid:
            raise ValueError("Geometry repair unsuccessful")

        glid_polygons.append(geom)

    except Exception as exc:
        invalid_records.append({
            "feature_index": i,
            "error": str(exc)
        })

pd.DataFrame(
    invalid_records,
    columns=["feature_index", "error"]
).to_csv(
    OUT / "glid_invalid_geometries.csv",
    index=False
)

print("Total shapefile features:", len(reader), flush=True)
print("Usable geometries:", len(glid_polygons), flush=True)
print("Excluded geometries:", len(invalid_records), flush=True)

if invalid_records:
    print("WARNING: Some GLID features could not be processed.")
    print("Their intersections remain unknown.", flush=True)

df = pd.read_csv(NEPAL / "metadata.csv")

results = []

for _, rec in df.iterrows():
    with rasterio.open(NEPAL / rec["GEOTIFF"]) as ds:
        footprint = box(*ds.bounds)

        transformer = Transformer.from_crs(
            ds.crs,
            "EPSG:4326",
            always_xy=True
        )

        footprint_wgs84 = transform(
            transformer.transform,
            footprint
        )

    intersecting = [
        i for i, polygon in enumerate(glid_polygons)
        if polygon.intersects(footprint_wgs84)
    ]

    results.append({
        "BASIN": rec["BASIN"],
        "GLO_ID": rec["GLO_ID"],
        "STAC_ID": rec["STAC_ID"],
        "GLID_POLYGON_INTERSECTIONS": len(intersecting),
        "HAS_GLID_POLYGON_OVERLAP": int(len(intersecting) > 0),
    })

    print(
        rec["BASIN"],
        rec["GLO_ID"],
        "GLID polygons:",
        len(intersecting),
        flush=True
    )

result_df = pd.DataFrame(results)

path = OUT / "glid_nepal_annotation_intersections.csv"
result_df.to_csv(path, index=False)

print("\n=== SUMMARY ===")
print("Nepal patches:", len(result_df))
print(
    "Patches intersecting GLID annotations:",
    result_df.HAS_GLID_POLYGON_OVERLAP.sum()
)
print(
    "Total polygon-patch intersections:",
    result_df.GLID_POLYGON_INTERSECTIONS.sum()
)

print("\n=== BY BASIN ===")
print(
    result_df.groupby("BASIN")[
        "HAS_GLID_POLYGON_OVERLAP"
    ].sum()
)

print("\nSaved:", path)
