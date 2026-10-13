import csv
import os
from pathlib import Path
from pystac_client import Client

root = Path("/tmp") / os.environ["USER"] / "glacial_lakes/nepal_glo"

targets = [
    ("Gandaki06", 83.50758, 28.81072),
    ("Karnali03", 82.03939, 30.31141),
    ("Koshi04", 87.81105, 27.91592),
]

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

rows = []

for name, lon, lat in targets:
    print(f"\n=== {name} ===", flush=True)

    items = list(catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[lon-0.015, lat-0.015, lon+0.015, lat+0.015],
        datetime="2023-09-01/2023-11-30",
        max_items=300
    ).items())

    candidates = {}

    for item in items:
        tile = item.properties.get("s2:mgrs_tile", "UNKNOWN")
        cloud = item.properties.get("eo:cloud_cover", 100)

        key = (tile, item.datetime.date().isoformat())

        if key not in candidates or cloud < candidates[key][0]:
            candidates[key] = (cloud, item.id)

    for (tile, date), (cloud, stac_id) in sorted(
        candidates.items(),
        key=lambda x: x[1][0]
    ):
        print(
            f"Tile={tile} Date={date} Cloud={cloud:.3f}",
            flush=True
        )

        rows.append({
            "LOCATION": name,
            "LONGITUDE": lon,
            "LATITUDE": lat,
            "TILE": tile,
            "DATE": date,
            "CLOUD_COVER": cloud,
            "STAC_ID": stac_id,
        })

out = Path("results/nepal_qa/neighbor_tile_candidates.csv")

with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "LOCATION", "LONGITUDE", "LATITUDE",
        "TILE", "DATE", "CLOUD_COVER", "STAC_ID"
    ])
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", out)
