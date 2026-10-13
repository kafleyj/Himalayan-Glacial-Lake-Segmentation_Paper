import pandas as pd
from pathlib import Path

src = Path("results/nepal_90_scene_candidates.csv")
out = Path("results/nepal_30_selected_scenes.csv")

df = pd.read_csv(src)
df["CLOUD_COVER"] = pd.to_numeric(
    df["CLOUD_COVER"], errors="coerce"
)

# Lowest scene-wide cloud-cover candidate per lake.
best = (
    df.sort_values(["CLOUD_COVER", "DATE"])
      .drop_duplicates("GLO_ID")
)

# Exclude the three previously evaluated pilot lakes.
pilots = {
    "GLO_83.52813_28.88699",
    "GLO_82.42308_29.38404",
    "GLO_86.69911_27.94145",
}
best = best[~best["GLO_ID"].isin(pilots)]

selected = []

for basin in ("Gandaki", "Karnali", "Koshi"):
    subset = best[best["BASIN"] == basin].copy()

    # Include different lake sizes where available.
    buckets = {
        size: subset[subset["size_group"] == size]
        .sort_values(["CLOUD_COVER", "GLO_ID"])
        for size in ("small", "medium", "large")
    }

    chosen = []
    used = set()

    for size, quota in (
        ("small", 3),
        ("medium", 3),
        ("large", 4),
    ):
        for _, row in buckets[size].head(quota).iterrows():
            chosen.append(row)
            used.add(row["GLO_ID"])

    if len(chosen) < 10:
        for _, row in subset.sort_values(
            ["CLOUD_COVER", "GLO_ID"]
        ).iterrows():
            if row["GLO_ID"] not in used:
                chosen.append(row)
                used.add(row["GLO_ID"])
            if len(chosen) == 10:
                break

    assert len(chosen) == 10, (
        f"Only {len(chosen)} selected for {basin}"
    )
    selected.extend(chosen)

result = pd.DataFrame(selected)
assert len(result) == 30
assert result["GLO_ID"].nunique() == 30

result.to_csv(out, index=False)

print("=== SELECTED NEPAL SCENES ===")
print(pd.crosstab(result["BASIN"], result["size_group"]))
print("\nCloud-cover summary (%):")
print(result.groupby("BASIN")["CLOUD_COVER"].agg(
    ["min", "mean", "max"]
).round(2))
print("\nUnique lakes:", result["GLO_ID"].nunique())
print("Unique scenes:", result["STAC_ID"].nunique())
print("Saved:", out)
print("\nNOTE: Selection is provisional; image QA is still required.")
