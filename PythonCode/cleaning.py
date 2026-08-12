import os
import pandas as pd
from PIL import Image

# =========================
# PATHS
# =========================

csv_path = "/home/gpuvm/Desktop/Luca Migliaccio/archive/Data_Entry_2017.csv"
image_root = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

# =========================
# LOAD CSV
# =========================

df = pd.read_csv(csv_path)

print("\n========================")
print("NIH DATASET CLEANING")
print("========================")

print("Total rows:", len(df))

# =========================
# FIND IMAGE PATH
# =========================

def get_image_path(img_name):

    for i in range(1, 13):

        folder = f"images_{i:03d}"

        path = os.path.join(
            image_root,
            folder,
            "images",
            img_name
        )

        if os.path.exists(path):
            return path

    return None


# =========================
# CLEANING STATS
# =========================

missing_images = []
corrupted_images = []

image_sizes = []
image_modes = {}

valid_images = 0

# =========================
# MAIN VALIDATION LOOP
# =========================

for idx, row in df.iterrows():

    img_name = row["Image Index"]

    img_path = get_image_path(img_name)

    # -------------------------
    # MISSING FILE
    # -------------------------

    if img_path is None:
        missing_images.append(img_name)
        continue

    # -------------------------
    # TRY OPEN IMAGE
    # -------------------------

    try:

        img = Image.open(img_path)

        # verify corruption
        img.verify()

        # reopen after verify
        img = Image.open(img_path)

        width, height = img.size
        image_sizes.append((width, height))

        # mode statistics
        mode = img.mode

        if mode not in image_modes:
            image_modes[mode] = 0

        image_modes[mode] += 1

        valid_images += 1

    except Exception as e:

        corrupted_images.append((img_name, str(e)))

    # -------------------------
    # PROGRESS PRINT
    # -------------------------

    if idx % 10000 == 0:

        print(f"Processed: {idx}/{len(df)}")


# =========================
# FINAL REPORT
# =========================

print("\n========================")
print("FINAL CLEANING REPORT")
print("========================")

print(f"Valid images: {valid_images}")
print(f"Missing images: {len(missing_images)}")
print(f"Corrupted images: {len(corrupted_images)}")

# =========================
# IMAGE SIZE STATS
# =========================

if len(image_sizes) > 0:

    widths = [s[0] for s in image_sizes]
    heights = [s[1] for s in image_sizes]

    print("\n========================")
    print("IMAGE SIZE STATS")
    print("========================")

    print(f"Min width: {min(widths)}")
    print(f"Max width: {max(widths)}")

    print(f"Min height: {min(heights)}")
    print(f"Max height: {max(heights)}")

# =========================
# IMAGE MODES
# =========================

print("\n========================")
print("IMAGE MODES")
print("========================")

for k, v in image_modes.items():
    print(f"{k}: {v}")

# =========================
# OPTIONAL DEBUG OUTPUTS
# =========================

if len(missing_images) > 0:

    print("\nFirst 10 missing images:")
    print(missing_images[:10])

if len(corrupted_images) > 0:

    print("\nFirst 10 corrupted images:")
    print(corrupted_images[:10])

print("\nCLEANING COMPLETED ✔")