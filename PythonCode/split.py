import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================

csv_path = "/home/gpuvm/Desktop/Luca Migliaccio/archive/Data_Entry_2017.csv"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =========================
# LOAD CSV
# =========================

df = pd.read_csv(csv_path)

print("\n========================")
print("PATIENT-LEVEL SPLIT")
print("========================")

print("Original dataset size:", len(df))

# =========================
# EXTRACT PATIENT ID
# =========================

def extract_patient_id(img_name):

    # Example:
    # 00000001_000.png
    # -> patient 00000001

    return img_name.split("_")[0]

df["PatientID"] = df["Image Index"].apply(extract_patient_id)

print("\nUnique patients:", df["PatientID"].nunique())

# =========================
# UNIQUE PATIENT LIST
# =========================

patients = df["PatientID"].unique().tolist()

random.seed(RANDOM_SEED)
random.shuffle(patients)

# =========================
# TRAIN / TEMP SPLIT
# =========================

train_patients, temp_patients = train_test_split(
    patients,
    test_size=(1 - TRAIN_RATIO),
    random_state=RANDOM_SEED
)

# =========================
# VAL / TEST SPLIT
# =========================

relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=relative_test_ratio,
    random_state=RANDOM_SEED
)

# =========================
# CREATE SPLITS
# =========================

train_df = df[df["PatientID"].isin(train_patients)]
val_df = df[df["PatientID"].isin(val_patients)]
test_df = df[df["PatientID"].isin(test_patients)]

print("\n========================")
print("LABEL DISTRIBUTION CHECK")
print("========================")

print(train_df["Finding Labels"].value_counts().head(10))

# =========================
# LEAKAGE CHECK
# =========================

train_ids = set(train_df["PatientID"])
val_ids = set(val_df["PatientID"])
test_ids = set(test_df["PatientID"])

train_val_overlap = train_ids.intersection(val_ids)
train_test_overlap = train_ids.intersection(test_ids)
val_test_overlap = val_ids.intersection(test_ids)

# =========================
# REPORT
# =========================

print("\n========================")
print("SPLIT REPORT")
print("========================")

print(f"Train images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Test images: {len(test_df)}")

print("\nPatient counts:")
print(f"Train patients: {len(train_ids)}")
print(f"Validation patients: {len(val_ids)}")
print(f"Test patients: {len(test_ids)}")

# =========================
# LEAKAGE REPORT
# =========================

print("\n========================")
print("LEAKAGE CHECK")
print("========================")

print("Train-Val overlap:", len(train_val_overlap))
print("Train-Test overlap:", len(train_test_overlap))
print("Val-Test overlap:", len(val_test_overlap))

# =========================
# SAVE CSV SPLITS
# =========================

save_dir = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"

train_df.to_csv(os.path.join(save_dir, "train_split.csv"), index=False)
val_df.to_csv(os.path.join(save_dir, "val_split.csv"), index=False)
test_df.to_csv(os.path.join(save_dir, "test_split.csv"), index=False)

print("\nCSV splits saved ✔")

print("\nTRAIN SPLIT EXAMPLE:")
print(train_df.head())

print("\nSPLIT COMPLETED ✔")