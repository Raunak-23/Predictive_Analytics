import zipfile
import io
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "bank_marketing.zip"

# Choose which dataset to process:
# "bank-full.csv"       -> classic Bank Marketing dataset
# "bank-additional-full.csv" -> extended version
DATASET_NAME = "bank-full.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "bank_marketing_output"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# 1. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. EXTRACT THE REQUIRED CSV FROM NESTED ZIP FILES
# ============================================================

def extract_dataset(zip_path, dataset_name):
    """
    Extract the requested CSV from the nested bank-marketing ZIP.
    """

    with zipfile.ZipFile(zip_path, "r") as outer_zip:

        # Search inside the outer ZIP
        for outer_file in outer_zip.namelist():

            if not outer_file.endswith(".zip"):
                continue

            nested_data = outer_zip.read(outer_file)

            with zipfile.ZipFile(io.BytesIO(nested_data), "r") as nested_zip:

                for file_name in nested_zip.namelist():

                    # Match the requested dataset
                    if file_name.endswith(dataset_name):

                        print(f"Found dataset: {file_name}")

                        csv_bytes = nested_zip.read(file_name)

                        return io.BytesIO(csv_bytes)

    raise FileNotFoundError(
        f"Could not find '{dataset_name}' inside {zip_path}"
    )


# ============================================================
# 3. LOAD DATASET
# ============================================================

csv_file = extract_dataset(ZIP_PATH, DATASET_NAME)

# UCI Bank Marketing uses ';' as the separator
df = pd.read_csv(
    csv_file,
    sep=";",
    encoding="utf-8"
)

print("\nOriginal dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 4. BASIC CLEANING
# ============================================================

# Remove accidental leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Remove leading/trailing spaces from string values
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()


# ------------------------------------------------------------
# Convert "unknown" to missing values
# ------------------------------------------------------------
# In this dataset, "unknown" represents an unavailable value
# rather than a meaningful category in most columns.

df = df.replace("unknown", pd.NA)


# ============================================================
# 5. REMOVE DUPLICATE RECORDS
# ============================================================

duplicates = df.duplicated().sum()

print(f"\nDuplicate rows found: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates().reset_index(drop=True)

print("Shape after duplicate removal:")
print(df.shape)


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print(df.isna().sum())


# ============================================================
# 7. TARGET VARIABLE
# ============================================================

TARGET = "y"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

# Normalize target
df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.lower()
    .str.strip()
)

# Remove invalid target rows, if any
df = df[df[TARGET].isin(["yes", "no"])].reset_index(drop=True)


# ============================================================
# 8. DATASET SUMMARY
# ============================================================

print("\nTarget distribution:")
print(df[TARGET].value_counts())

print("\nTarget distribution (%):")
print(
    df[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 9. SAVE CLEAN COMPLETE DATASET
# ============================================================

clean_path = OUTPUT_DIR / "bank_marketing_clean.csv"

df.to_csv(
    clean_path,
    index=False,
    encoding="utf-8"
)

print(f"\nClean dataset saved to:")
print(clean_path)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# 11. RECOMBINE X AND y
# ============================================================

train_df = X_train.copy()
train_df[TARGET] = y_train

test_df = X_test.copy()
test_df[TARGET] = y_test


# Reset indices
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# 12. SAVE TRAIN / TEST CSV FILES
# ============================================================

train_path = OUTPUT_DIR / "train.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(
    train_path,
    index=False,
    encoding="utf-8"
)

test_df.to_csv(
    test_path,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 13. SAVE DATASET INFORMATION
# ============================================================

info_path = OUTPUT_DIR / "dataset_info.txt"

with open(info_path, "w", encoding="utf-8") as f:

    f.write("BANK MARKETING DATASET\n")
    f.write("======================\n\n")

    f.write(f"Source dataset: {DATASET_NAME}\n")
    f.write(f"Total records: {len(df)}\n")
    f.write(f"Total features: {len(df.columns) - 1}\n")
    f.write(f"Target: {TARGET}\n\n")

    f.write("Target distribution:\n")
    f.write(str(df[TARGET].value_counts()))
    f.write("\n\n")

    f.write("Target distribution (%):\n")
    f.write(
        str(
            df[TARGET]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )
    )

    f.write("\n\n")
    f.write("Feature data types:\n")
    f.write(str(df.dtypes))

    f.write("\n\n")
    f.write("Missing values:\n")
    f.write(str(df.isna().sum()))

    f.write("\n\n")
    f.write(f"Training samples: {len(train_df)}\n")
    f.write(f"Testing samples: {len(test_df)}\n")


# ============================================================
# 14. FINAL SUMMARY
# ============================================================
print(f"Total samples : {len(df)}")
print(f"Train samples : {len(train_df)}")
print(f"Test samples  : {len(test_df)}")
print(f"Features      : {len(df.columns) - 1}")
print(f"Target        : {TARGET}")

print("\nGenerated files:")
print(f"  {clean_path}")
print(f"  {train_path}")
print(f"  {test_path}")
print(f"  {info_path}")