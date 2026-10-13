import pandas as pd
from pathlib import Path

project = Path.home() / "Himalayan_Glacial_Lake_Segmentation"
src = project / "results/nepal_negative_candidates.csv"

df = pd.read_csv(src)

review = {
    "Gandaki_NEG_02": "REVIEW",
    "Gandaki_NEG_05": "REVIEW",
    "Gandaki_NEG_10": "REVIEW",
    "Karnali_NEG_08": "REVIEW",
    "Karnali_NEG_09": "REVIEW",
    "Koshi_NEG_02": "REVIEW",
    "Koshi_NEG_03": "REVIEW",
    "Koshi_NEG_04": "REVIEW",
    "Koshi_NEG_05": "REVIEW",
    "Koshi_NEG_06": "REVIEW",
}

df["QA_STATUS"] = df["NEG_ID"].map(
    lambda x: review.get(x, "PROVISIONAL_PASS")
)

df["QA_METHOD"] = "Contact-sheet visual inspection"
df["QA_NOTE"] = df["NEG_ID"].map(
    lambda x: (
        "Possible water or unsuitable terrain; inspect at native resolution"
        if x in review
        else "No obvious lake at thumbnail scale; requires full-resolution QA"
    )
)

out = project / "results/nepal_negative_manual_review.csv"
df.to_csv(out, index=False)

print(df.groupby(["BASIN", "QA_STATUS"]).size())
print("\nSaved:", out)
