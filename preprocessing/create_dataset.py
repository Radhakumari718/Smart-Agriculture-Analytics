import os
import csv
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

suffixes = {
    "area": "_HarvestedAreaHectares.tif",
    "yield": "_YieldPerHectare.tif",
    "production": "_Production.tif"
}

step = 20
crops = []

for folder in sorted(os.listdir(".")):
    if not os.path.isdir(folder) or folder == "__MACOSX":
        continue
    if all(os.path.exists(os.path.join(folder, folder + s))
           for s in suffixes.values()):
        crops.append(folder)

print("Total crops:", len(crops))

first = gdal.Open(os.path.join(crops[0], crops[0] + suffixes["area"]))
gt = first.GetGeoTransform()
first = None

with open("smart_agriculture_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["latitude", "longitude"]
    for crop in crops:
        for feature in suffixes:
            header.append(crop + "_" + feature)
    writer.writerow(header)

    for row in range(0, 2160, step):
        datasets = {}
        for crop in crops:
            datasets[crop] = {}
            for feature, suffix in suffixes.items():
                path = os.path.join(crop, crop + suffix)
                ds = gdal.Open(path)
                arr = ds.GetRasterBand(1).ReadAsArray(
                    0, row, 4320, min(step, 2160-row)
                )
                datasets[crop][feature] = arr
                ds = None

        for col in range(0, 4320, step):
            values = []
            for crop in crops:
                for feature in suffixes:
                    value = datasets[crop][feature][0, col]
                    values.append(float(value) if np.isfinite(value) else 0)

            lat = gt[3] + (row + 0.5) * gt[5]
            lon = gt[0] + (col + 0.5) * gt[1]
            writer.writerow([lat, lon] + values)

        print("Processed row:", row)

print("CSV CREATED SUCCESSFULLY!")
print("Total crops:", len(crops))
