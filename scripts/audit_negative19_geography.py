import os
import sys
import csv
import importlib.util
from pathlib import Path

import pandas as pd
import rasterio
import shapefile
import planetary_computer
from pystac_client import Client
from pyproj import Transformer
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform, unary_union

PROJECT = Path.home() / "Himalayan_Glacial_Lake_Segmentation"
ROOT = Path("/tmp") / os.environ["USER"] / "glacial_lakes"
NEPAL_ROOT = ROOT / "nepal_glo"
OUT = PROJECT / "results/nepal_geospatial_audit"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    PROJECT / "results/nepal_negative_provisional19.csv"
)

assert len(df) == 19
assert df["NEG_ID"].is_unique
assert (df["QA_STATUS"] == "PROVISIONAL_PASS").all()

os.chdir(NEPAL_ROOT)
spec = importlib.util.spec_from_file_location(
    "pilot_builder",
    NEPAL_ROOT / "build_nepal_pilot_dataset.py"
)
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)

nepal_polygons = builder.load_nepal_polygons()
reader = shapefile.Reader(
    str(ROOT / "extracted/annotations/GLID_annotation.shp")
)

glid_geometries = []
invalid_glid = 0

for feature in reader.iterShapeRecords():
    try:
        geom = shape(feature.shape.__geo_interface__)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty or not geom.is_valid:
            raise ValueError("Unusable geometry")
        glid_geometries.append(geom)
    except Exception:
        invalid_glid += 1

print("Nepal inventory polygons:", len(nepal_polygons), flush=True)
print("GLID usable geometries:", len(glid_geometries), flush=True)
print("GLID excluded geometries:", invalid_glid, flush=True)

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

scene_cache = {}
inventory_cache = {}
rows = []

for _, rec in df.iterrows():
    scene_id = str(rec["STAC_ID"])

    if scene_id not in scene_cache:
        items = list(catalog.search(
            collections=["sentinel-2-l2a"],
            ids=[scene_id]
        ).items())

        if not items:
            raise RuntimeError(f"Scene not found: {scene_id}")

        scene_cache[scene_id] = planetary_computer.sign(items[0])

    item = scene_cache[scene_id]

    with rasterio.open(item.assets["B04"].href) as ds:
        to_scene = Transformer.from_crs(
            "EPSG:4326", ds.crs, always_xy=True
        )

        x, y = to_scene.transform(
            float(rec["LONGITUDE"]),
            float(rec["LATITUDE"])
        )

        row, col = ds.index(x, y)

        window = Window(
            col_off=col - 128,
            row_off=row - 128,
            width=256,
            height=256
        )

        footprint = box(*window_bounds(window, ds.transform))
        crs = ds.crs

        key = str(crs)

        if key not in inventory_cache:
            convert = Transformer.from_crs(
                "ESRI:102025", crs, always_xy=True
            ).transform

            projected = [
                shp_transform(convert, poly["geometry"])
                for poly in nepal_polygons
                if poly["geometry"] is not None
                and not poly["geometry"].is_empty
            ]

            inventory_cache[key] = unary_union(projected)

        nepal_overlap = inventory_cache[key].intersects(
            footprint.buffer(100)
        )

        to_wgs84 = Transformer.from_crs(
            crs, "EPSG:4326", always_xy=True
        ).transform

        footprint_wgs84 = shp_transform(
            to_wgs84, footprint
        )

        glid_count = sum(
            geom.intersects(footprint_wgs84)
            for geom in glid_geometries
        )

        rows.append({
            "BASIN": rec["BASIN"],
            "NEG_ID": rec["NEG_ID"],
            "STAC_ID": scene_id,
            "CRS": str(crs),
            "NEPAL_INVENTORY_OVERLAP_100M": int(nepal_overlap),
            "GLID_ANNOTATION_INTERSECTIONS": glid_count,
            "QA_STATUS": "PENDING_FINAL_VERIFICATION",
        })

        print(
            f"{rec['NEG_ID']}: "
            f"Nepal_overlap={int(nepal_overlap)} "
            f"GLID_intersections={glid_count}",
            flush=True
        )

result = pd.DataFrame(rows)

path = OUT / "negative19_polygon_audit.csv"
result.to_csv(path, index=False)

print("\n=== AUDIT SUMMARY ===")
print("Samples:", len(result))
print(
    "Nepal inventory overlaps:",
    int(result.NEPAL_INVENTORY_OVERLAP_100M.sum())
)
print(
    "Samples intersecting GLID annotations:",
    int((result.GLID_ANNOTATION_INTERSECTIONS > 0).sum())
)
print("Excluded GLID geometries:", invalid_glid)
print("Saved:", path)
