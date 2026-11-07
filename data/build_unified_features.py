
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

LOGIN_FILE = DATA_DIR / "daily_activity_features.csv"
SUMMARY_FILE = DATA_DIR / "daily_activity_summary.csv"
OUTPUT_FILE = DATA_DIR / "unified_activity_features.csv"

print("Loading datasets...")

login = pd.read_csv(
    LOGIN_FILE,
    dtype={"user": "string"},
    low_memory=False
)

summary = pd.read_csv(
    SUMMARY_FILE,
    dtype={"user": "string"},
    low_memory=False
)

login_required = {"user", "calendar_date"}
summary_required = {"user", "date"}

if not login_required.issubset(login.columns):
    raise ValueError("Login dataset is missing required columns.")

if not summary_required.issubset(summary.columns):
    raise ValueError("Summary dataset is missing required columns.")

login["calendar_date"] = pd.to_datetime(
    login["calendar_date"],
    errors="coerce"
).dt.normalize()

summary["calendar_date"] = pd.to_datetime(
    summary["date"],
    errors="coerce"
).dt.normalize()

login = login.drop(columns=["date"], errors="ignore")
summary = summary.drop(columns=["date"])

if login[["user", "calendar_date"]].isna().any().any():
    raise ValueError("Login dataset contains invalid employee-date keys.")

if summary[["user", "calendar_date"]].isna().any().any():
    raise ValueError("Summary dataset contains invalid employee-date keys.")

if login.duplicated(["user", "calendar_date"]).any():
    raise ValueError("Duplicate employee-date keys in login dataset.")

if summary.duplicated(["user", "calendar_date"]).any():
    raise ValueError("Duplicate employee-date keys in summary dataset.")

if "unique_pcs" in summary.columns:
    summary = summary.rename(
        columns={"unique_pcs": "summary_unique_pcs"}
    )

print("Merging datasets...")

unified = login.merge(
    summary,
    on=["user", "calendar_date"],
    how="left",
    validate="one_to_one",
    indicator=True
)

unmatched = (unified["_merge"] != "both").sum()

if unmatched > 0:
    raise ValueError(
        f"{unmatched} login records have no matching summary record."
    )

unified = unified.drop(columns=["_merge"])

unified = unified.sort_values(
    ["user", "calendar_date"]
).reset_index(drop=True)

if unified.duplicated(["user", "calendar_date"]).any():
    raise ValueError("Duplicate keys detected after merging.")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

unified.to_csv(OUTPUT_FILE, index=False)

print("\nUNIFIED DATASET CREATED")
print("Output:", OUTPUT_FILE)
print("Records:", len(unified))
print("Employees:", unified["user"].nunique())
print("Columns:", len(unified.columns))
print("Date range:", unified["calendar_date"].min(),
      "to", unified["calendar_date"].max())
print("\nAvailable columns:")
print(unified.columns.tolist())
