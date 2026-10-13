import os
from pathlib import Path
import shapefile

root = (
    Path("/tmp") / os.environ["USER"]
    / "glacial_lakes/extracted/annotations"
)

path = root / "GLID_annotation.shp"

reader = shapefile.Reader(str(path))

print("=== GLID SHAPEFILE ===")
print("Number of features:", len(reader))
print("Bounding box:", reader.bbox)
print("Geometry type:", reader.shapeTypeName)

print("\n=== ATTRIBUTE FIELDS ===")
for field in reader.fields[1:]:
    print(field)

print("\n=== FIRST 5 RECORDS ===")
for i, record in enumerate(reader.iterRecords()):
    if i >= 5:
        break
    print(i + 1, record.as_dict())

print("\n=== CRS INFORMATION ===")
prj = root / "GLID_annotation.prj"

if prj.exists():
    print(prj.read_text()[:2000])
else:
    print("Projection file unavailable.")

print("\nInspection completed.")
