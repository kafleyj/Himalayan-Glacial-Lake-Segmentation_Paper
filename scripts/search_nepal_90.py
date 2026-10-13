import csv
import os
import time
from pathlib import Path

from pystac_client import Client

root = Path("/tmp") / os.environ["USER"] / "glacial_lakes/nepal_glo"
candidates = root / "nepal_90_spatially_separated.csv"
output = Path("results/nepal_90_scene_candidates.csv")
output.parent.mkdir(parents=True, exist_ok=True)

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

with open(candidates, newline="") as f:
    lakes = list(csv.DictReader(f))

results = []
failed = []

for i, lake in enumerate(lakes, 1):
    lon = float(lake["LONGITUDE"])
    lat = float(lake["LATITUDE"])

    bbox = [
        lon - 0.02, lat - 0.02,
        lon + 0.02, lat + 0.02
    ]

    print(
        f"[{i}/{len(lakes)}] {lake['BASIN']} "
        f"{lake['GLO_ID']}",
        flush=True
    )

    try:
        items = list(
            catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime="2023-09-01/2023-11-30",
                query={"eo:cloud_cover": {"lt": 20}},
                max_items=50
            ).items()
        )

        items.sort(
            key=lambda item: (
                item.properties.get("eo:cloud_cover", 100),
                item.datetime.isoformat()
            )
        )

        # Store up to 3 scenes per lake for subsequent QA.
        for item in items[:3]:
            results.append({
                "GLO_ID": lake["GLO_ID"],
                "BASIN": lake["BASIN"],
                "LONGITUDE": lon,
                "LATITUDE": lat,
                "size_group": lake["size_group"],
                "STAC_ID": item.id,
                "DATE": item.datetime.date().isoformat(),
                "CLOUD_COVER": item.properties.get(
                    "eo:cloud_cover", 100
                ),
            })

        print(f"  Found {len(items)} scenes", flush=True)

    except Exception as exc:
        failed.append((lake["GLO_ID"], str(exc)))
        print("  ERROR:", exc, flush=True)

    time.sleep(0.2)

fields = [
    "GLO_ID", "BASIN", "LONGITUDE", "LATITUDE",
    "size_group", "STAC_ID", "DATE", "CLOUD_COVER"
]

with open(output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print("\n=== SEARCH SUMMARY ===")
print("Lakes searched:", len(lakes))
print("Candidate scenes:", len(results))
print("Lakes with scenes:", len(set(r["GLO_ID"] for r in results)))
print("Errors:", len(failed))
print("Saved:", output)

for lake_id, error in failed[:10]:
    print("FAILED:", lake_id, error)
