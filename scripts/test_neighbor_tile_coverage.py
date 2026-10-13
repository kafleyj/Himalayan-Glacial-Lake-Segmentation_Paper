import os
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from pystac_client import Client
import planetary_computer

project = Path.cwd()
root = Path("/tmp") / os.environ["USER"] / "glacial_lakes/nepal_glo"

df = pd.read_csv(
    project / "results/nepal_qa/neighbor_tile_candidates.csv"
)

targets = {
    "Gandaki06": "44RQS",
    "Karnali03": "44RNU",
    "Koshi04": "45RWL",
}

os.chdir(root)

spec = importlib.util.spec_from_file_location(
    "pilot_builder",
    root / "build_nepal_pilot_dataset.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

os.chdir(project)

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

results = []

for location, tile in targets.items():
    subset = df[
        (df.LOCATION == location) &
        (df.TILE.astype(str).str.replace("T", "", n=1) == tile)
    ].sort_values("CLOUD_COVER").head(5)

    print(f"\n=== {location} | {tile} ===", flush=True)

    for _, row in subset.iterrows():
        try:
            items = list(catalog.search(
                collections=["sentinel-2-l2a"],
                ids=[row.STAC_ID]
            ).items())

            if not items:
                raise RuntimeError("Scene not found")

            item = planetary_computer.sign(items[0])

            rgb, _, _, _ = builder.read_rgb_patch(
                item, float(row.LONGITUDE), float(row.LATITUDE)
            )

            coverage = float(
                np.any(rgb != 0, axis=0).mean()
            )

            print(
                row.DATE,
                row.TILE,
                f"coverage={coverage:.3f}",
                flush=True
            )

            results.append({
                **row.to_dict(),
                "RGB_COVERAGE": coverage,
                "STATUS": "OK"
            })

        except Exception as exc:
            print("ERROR:", repr(exc), flush=True)
            results.append({
                **row.to_dict(),
                "RGB_COVERAGE": None,
                "STATUS": repr(exc)
            })

out = project / "results/nepal_qa/neighbor_coverage.csv"
pd.DataFrame(results).to_csv(out, index=False)

print("\nSaved:", out)
