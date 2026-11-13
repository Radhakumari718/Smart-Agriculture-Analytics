import os
import csv
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

GEOTIFF_DIR = "/mnt/c/Users/anjal/Downloads/HarvestedAreaYield175Crops_Geotiff/HarvestedAreaYield175Crops_Geotiff/GeoTiff"
OUTPUT_CSV = os.path.expanduser("~/Smart-Agriculture-Analytics/dataset/smart_agriculture_dataset_corrected.csv")

suffixes = {
    "area": "_HarvestedAreaHectares.tif",
    "yield": "_YieldPerHectare.tif",
    "production": "_Production.tif"
}

step = 20
crops = []

for folder in sorted(os.listdir(GEOTIFF_DIR)):
    folder_path = os.path.join(GEOTIFF_DIR, folder)

    if not os.path.isdir(folder_path) or folder == "__MACOSX":
        continue

    if all(os.path.exists(os.path.join(folder_path, folder + s))
           for s in suffixes.values()):
        crops.append(folder)

print("Total crops:", len(crops))

first = gdal.Open(os.path.join(
    GEOTIFF_DIR, crops[0], crops[0] + suffixes["area"]
))
gt = first.GetGeoTransform()
first = None

header = ["latitude", "longitude"]

for crop in crops:
    for feature in suffixes:
        header.append(crop + "_" + feature)

fixed_chicory = 0
total_records = 0

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for row in range(0, 2160, step):
        datasets = {}

        for crop in crops:
            datasets[crop] = {}

            for feature, suffix in suffixes.items():
                path = os.path.join(
                    GEOTIFF_DIR, crop, crop + suffix
                )

                ds = gdal.Open(path)
                band = ds.GetRasterBand(1)

                arr = band.ReadAsArray(
                    0, row, 4320, min(step, 2160 - row)
                )

                nodata = band.GetNoDataValue()

                datasets[crop][feature] = (arr, nodata)
                ds = None

        for col in range(0, 4320, step):
            values = []

            for crop in crops:
                crop_values = {}

                for feature in suffixes:
                    arr, nodata = datasets[crop][feature]
                    value = float(arr[0, col])

                    if not np.isfinite(value):
                        value = 0

                    elif nodata is not None and value == nodata:
                        value = 0

                    crop_values[feature] = value

                if (
                    crop == "chicory"
                    and crop_values["production"] > 1e9
                    and crop_values["yield"] == 0
                ):
                    crop_values["production"] = 0
                    fixed_chicory += 1

                for feature in suffixes:
                    values.append(crop_values[feature])

            lat = gt[3] + (row + 0.5) * gt[5]
            lon = gt[0] + (col + 0.5) * gt[1]

            writer.writerow([lat, lon] + values)
            total_records += 1

        print("Processed row:", row)

print("\nCSV CREATED SUCCESSFULLY!")
print("Total crops:", len(crops))
print("Total records:", total_records)
print("Corrected Chicory anomalies in sampled data:", fixed_chicory)
print("Output:", OUTPUT_CSV)
