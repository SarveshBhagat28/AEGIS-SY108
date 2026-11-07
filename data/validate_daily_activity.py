import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

DAILY_FILE = DATA_DIR / "daily_activity_summary.csv"

SOURCE_FILES = {
    "logon": DATA_DIR / "logon.csv",
    "file": DATA_DIR / "file.csv",
    "device": DATA_DIR / "device.csv"
}

CHUNK_SIZE = 200_000

print("=" * 65)
print("AEGIS - DAILY DATASET VALIDATION")
print("=" * 65)

# ---------------------------------------------------------
# 1. LOAD GENERATED DATASET
# ---------------------------------------------------------

daily = pd.read_csv(DAILY_FILE, parse_dates=["date"])

print("\n1. GENERATED DATASET")

print("Rows:", len(daily))
print("Unique employees:", daily["user"].nunique())

print("Duplicate user-date records:",
      daily.duplicated(["user", "date"]).sum())

print("Missing values:")
print(daily.isnull().sum().to_string())


# ---------------------------------------------------------
# 2. VALIDATE DAILY EVENT COUNTS AGAINST RAW FILES
# ---------------------------------------------------------

checks = {
    "logon": {
        "logon": "logon_events",
        "logoff": "logoff_events"
    },
    "file": {
        "file": "file_events",
        "copy": "file_copies",
        "open": "file_opens",
        "delete": "file_deletes",
        "write": "file_writes",
        "to_removable": "to_removable_events",
        "from_removable": "from_removable_events",
        "removable": "removable_media_events"
    },
    "device": {
        "file": "device_events",
        "connect": "device_connects",
        "disconnect": "device_disconnects"
    }
}

for source, file_path in SOURCE_FILES.items():

    print(f"\n2. VALIDATING {source.upper()}")

    actual = {}

    if source == "logon":
        usecols = ["activity"]

    elif source == "file":
        usecols = [
            "activity",
            "to_removable_media",
            "from_removable_media"
        ]

    else:
        usecols = ["activity"]

    for chunk in pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=CHUNK_SIZE
    ):

        activity = chunk["activity"].astype(str).str.strip().str.lower()

        if source == "logon":

            actual["logon"] = actual.get("logon", 0) + (activity == "logon").sum()
            actual["logoff"] = actual.get("logoff", 0) + (activity == "logoff").sum()

        elif source == "file":

            actual["file"] = actual.get("file", 0) + len(chunk)

            for event in ["copy", "open", "delete", "write"]:
                actual[event] = actual.get(event, 0) + (
                    activity == f"file {event}"
                ).sum()

            to_removable = (
                chunk["to_removable_media"]
                .astype(str).str.strip().str.lower()
                .isin(["true", "1", "yes"])
            )

            from_removable = (
                chunk["from_removable_media"]
                .astype(str).str.strip().str.lower()
                .isin(["true", "1", "yes"])
            )

            actual["to_removable"] = actual.get("to_removable", 0) + to_removable.sum()
            actual["from_removable"] = actual.get("from_removable", 0) + from_removable.sum()
            actual["removable"] = actual.get("removable", 0) + (
                to_removable | from_removable
            ).sum()

        else:

            actual["file"] = actual.get("file", 0) + len(chunk)
            actual["connect"] = actual.get("connect", 0) + (activity == "connect").sum()
            actual["disconnect"] = actual.get("disconnect", 0) + (activity == "disconnect").sum()

    daily_columns = checks[source]

    for raw_name, daily_column in daily_columns.items():

        expected = actual.get(raw_name, 0)
        observed = int(daily[daily_column].sum())

        status = "PASS" if expected == observed else "FAIL"

        print(
            f"{daily_column:28} "
            f"Raw: {expected:12,} "
            f"Daily: {observed:12,} "
            f"{status}"
        )


# ---------------------------------------------------------
# 3. COVERAGE VALIDATION
# ---------------------------------------------------------

print("\n3. COVERAGE VALIDATION")

for source in ["logon", "file", "device"]:

    column = f"{source}_coverage"

    print(
        f"{source.capitalize():10} "
        f"Employees: {daily.loc[daily[column], 'user'].nunique():,}"
    )


# ---------------------------------------------------------
# 4. DATE AND ACTIVITY VALIDATION
# ---------------------------------------------------------

print("\n4. DATE AND ACTIVITY VALIDATION")

print("Earliest date:", daily["date"].min())
print("Latest date:", daily["date"].max())

print("Negative event counts:",
      (daily.select_dtypes(include="number") < 0).sum().sum())

print("Rows with zero total events:",
      (daily["total_events"] == 0).sum())

calculated_total = (
    daily["logon_events"]
    + daily["logoff_events"]
    + daily["file_events"]
    + daily["device_events"]
)

print(
    "Total event calculation:",
    "PASS" if (calculated_total == daily["total_events"]).all()
    else "FAIL"
)


# ---------------------------------------------------------
# 5. FINAL RESULT
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("VALIDATION COMPLETED")
print("=" * 65)