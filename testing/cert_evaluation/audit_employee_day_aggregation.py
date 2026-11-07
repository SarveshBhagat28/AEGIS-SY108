import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "testing" / "reports"

CHUNK_SIZE = 200_000

INSIDERS = [
    "ACM2278",
    "CMP2946",
    "PLJ1771",
    "CDE1846",
    "MBG3183",
]

DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

COUNT_COLUMNS = [
    "logon_events",
    "logoff_events",
    "file_events",
    "file_copies",
    "file_opens",
    "file_deletes",
    "file_writes",
    "to_removable_events",
    "from_removable_events",
    "removable_media_events",
    "device_events",
    "device_connects",
    "device_disconnects",
]

COMPARISON_COLUMNS = COUNT_COLUMNS + ["total_events"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_activity(series):
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
    )


def normalize_boolean(series):
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def prepare_chunk(df):

    df["date"] = pd.to_datetime(
        df["date"],
        format=DATE_FORMAT,
        errors="coerce"
    )

    df = df.dropna(
        subset=["user", "date"]
    ).copy()

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    return df


# ============================================================
# RAW DATA AGGREGATION
# ============================================================

def aggregate_source(filename, source_type):

    filepath = DATA_DIR / filename

    partial_results = []

    print(f"\nProcessing {filename}...")

    for chunk in pd.read_csv(
        filepath,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        # Restrict processing to the five CERT insiders.
        chunk = chunk[
            chunk["user"].isin(INSIDERS)
        ].copy()

        if chunk.empty:
            continue

        chunk = prepare_chunk(chunk)

        if chunk.empty:
            continue

        # ----------------------------------------------------
        # LOGON
        # ----------------------------------------------------

        if source_type == "logon":

            activity = normalize_activity(
                chunk["activity"]
            )

            chunk["logon_events"] = (
                activity == "logon"
            ).astype(int)

            chunk["logoff_events"] = (
                activity == "logoff"
            ).astype(int)

            grouped = chunk.groupby(
                ["user", "date"],
                as_index=False
            ).agg(
                logon_events=("logon_events", "sum"),
                logoff_events=("logoff_events", "sum")
            )

        # ----------------------------------------------------
        # FILE
        # ----------------------------------------------------

        elif source_type == "file":

            activity = normalize_activity(
                chunk["activity"]
            )

            chunk["file_events"] = 1

            chunk["file_copies"] = (
                activity == "file copy"
            ).astype(int)

            chunk["file_opens"] = (
                activity == "file open"
            ).astype(int)

            chunk["file_deletes"] = (
                activity == "file delete"
            ).astype(int)

            chunk["file_writes"] = (
                activity == "file write"
            ).astype(int)

            chunk["to_removable_events"] = (
                normalize_boolean(
                    chunk["to_removable_media"]
                )
            ).astype(int)

            chunk["from_removable_events"] = (
                normalize_boolean(
                    chunk["from_removable_media"]
                )
            ).astype(int)

            chunk["removable_media_events"] = (
                (
                    chunk["to_removable_events"]
                    + chunk["from_removable_events"]
                ) > 0
            ).astype(int)

            grouped = chunk.groupby(
                ["user", "date"],
                as_index=False
            ).agg(
                file_events=("file_events", "sum"),
                file_copies=("file_copies", "sum"),
                file_opens=("file_opens", "sum"),
                file_deletes=("file_deletes", "sum"),
                file_writes=("file_writes", "sum"),
                to_removable_events=(
                    "to_removable_events", "sum"
                ),
                from_removable_events=(
                    "from_removable_events", "sum"
                ),
                removable_media_events=(
                    "removable_media_events", "sum"
                )
            )

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        elif source_type == "device":

            activity = normalize_activity(
                chunk["activity"]
            )

            chunk["device_events"] = 1

            chunk["device_connects"] = (
                activity == "connect"
            ).astype(int)

            chunk["device_disconnects"] = (
                activity == "disconnect"
            ).astype(int)

            grouped = chunk.groupby(
                ["user", "date"],
                as_index=False
            ).agg(
                device_events=("device_events", "sum"),
                device_connects=("device_connects", "sum"),
                device_disconnects=("device_disconnects", "sum")
            )

        else:
            raise ValueError(
                f"Unknown source type: {source_type}"
            )

        partial_results.append(grouped)

    if not partial_results:

        print("No records found.")

        return pd.DataFrame(
            columns=["user", "date"]
        )

    result = pd.concat(
        partial_results,
        ignore_index=True
    )

    value_columns = [
        column
        for column in result.columns
        if column not in ["user", "date"]
    ]

    result = result.groupby(
        ["user", "date"],
        as_index=False
    )[value_columns].sum()

    print(
        f"Employee-date records: {len(result):,}"
    )

    return result


# ============================================================
# 1. PROCESS RAW TELEMETRY
# ============================================================

print("=" * 75)
print("CERT EMPLOYEE-DAY AGGREGATION AUDIT")
print("=" * 75)

logon = aggregate_source(
    "logon.csv",
    "logon"
)

file_data = aggregate_source(
    "file.csv",
    "file"
)

device = aggregate_source(
    "device.csv",
    "device"
)


# ============================================================
# 2. MERGE RAW AGGREGATES
# ============================================================

raw_daily = logon.merge(
    file_data,
    on=["user", "date"],
    how="outer"
)

raw_daily = raw_daily.merge(
    device,
    on=["user", "date"],
    how="outer"
)

for column in COUNT_COLUMNS:

    if column not in raw_daily.columns:
        raw_daily[column] = 0

    raw_daily[column] = (
        raw_daily[column]
        .fillna(0)
        .astype(int)
    )

# Match the original build_daily_activity.py definition.
raw_daily["total_events"] = (
    raw_daily["logon_events"]
    + raw_daily["logoff_events"]
    + raw_daily["file_events"]
    + raw_daily["device_events"]
)


# ============================================================
# 3. LOAD GENERATED DAILY SUMMARY
# ============================================================

print("\nLoading daily_activity_summary.csv...")

summary = pd.read_csv(
    DATA_DIR / "daily_activity_summary.csv",
    low_memory=False
)

summary = summary[
    summary["user"].isin(INSIDERS)
].copy()

summary["date"] = pd.to_datetime(
    summary["date"],
    errors="coerce"
).dt.strftime("%Y-%m-%d")

summary = summary.dropna(
    subset=["user", "date"]
)

missing_columns = [
    column
    for column in COMPARISON_COLUMNS
    if column not in summary.columns
]

if missing_columns:
    raise ValueError(
        f"Summary is missing columns: {missing_columns}"
    )

summary = summary[
    ["user", "date"] + COMPARISON_COLUMNS
].copy()

for column in COMPARISON_COLUMNS:

    summary[column] = pd.to_numeric(
        summary[column],
        errors="coerce"
    ).fillna(0).astype(int)


# ============================================================
# 4. CHECK DUPLICATE EMPLOYEE-DATE RECORDS
# ============================================================

raw_duplicates = raw_daily.duplicated(
    subset=["user", "date"]
).sum()

summary_duplicates = summary.duplicated(
    subset=["user", "date"]
).sum()

if raw_duplicates or summary_duplicates:

    raise ValueError(
        "Duplicate employee-date records detected. "
        f"Raw: {raw_duplicates}, Summary: {summary_duplicates}"
    )


# ============================================================
# 5. COMPARE RAW AND GENERATED DATA
# ============================================================

comparison = raw_daily.merge(
    summary,
    on=["user", "date"],
    how="outer",
    suffixes=("_raw", "_summary"),
    indicator=True
)

comparison["record_status"] = comparison["_merge"].map({
    "both": "MATCHED_EMPLOYEE_DAY",
    "left_only": "MISSING_FROM_SUMMARY",
    "right_only": "MISSING_FROM_RAW"
})

mismatch_columns = []

for column in COMPARISON_COLUMNS:

    raw_col = f"{column}_raw"
    summary_col = f"{column}_summary"

    comparison[raw_col] = (
        comparison[raw_col]
        .fillna(0)
        .astype(int)
    )

    comparison[summary_col] = (
        comparison[summary_col]
        .fillna(0)
        .astype(int)
    )

    mismatch_col = f"{column}_mismatch"

    comparison[mismatch_col] = (
        comparison[raw_col]
        != comparison[summary_col]
    )

    mismatch_columns.append(mismatch_col)

comparison["any_count_mismatch"] = (
    comparison[mismatch_columns]
    .any(axis=1)
)


# ============================================================
# 6. EXPORT DETAILED AUDIT REPORT
# ============================================================

report_columns = [
    "user",
    "date",
    "record_status",
    "any_count_mismatch"
]

for column in COMPARISON_COLUMNS:

    report_columns.extend([
        f"{column}_raw",
        f"{column}_summary",
        f"{column}_mismatch"
    ])

report = comparison[
    report_columns
].copy()

report = report.sort_values(
    ["user", "date"]
)

report_path = (
    REPORT_DIR
    / "employee_day_aggregation_audit.csv"
)

report.to_csv(
    report_path,
    index=False
)


# ============================================================
# 7. AUDIT SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("AUDIT RESULTS")
print("=" * 75)

print(
    "\nEmployees evaluated:",
    report["user"].nunique()
)

print(
    "Raw employee-day records:",
    len(raw_daily)
)

print(
    "Summary employee-day records:",
    len(summary)
)

print(
    "Compared employee-day records:",
    len(report)
)

print("\nRecord status:")

print(
    report["record_status"]
    .value_counts()
    .to_string()
)

print(
    "\nEmployee-day count mismatches:",
    int(report["any_count_mismatch"].sum())
)

print("\nMismatches by metric:")

for column in mismatch_columns:

    metric = column.replace(
        "_mismatch", ""
    )

    count = int(
        report[column].sum()
    )

    print(f"{metric}: {count}")


# ============================================================
# 8. EMPLOYEE-WISE RESULTS
# ============================================================

print("\nEmployee-wise results:")

for employee in INSIDERS:

    employee_report = report[
        report["user"] == employee
    ]

    print(f"\n{employee}")

    print(
        "Employee-days:",
        len(employee_report)
    )

    print(
        "Mismatched days:",
        int(
            employee_report[
                "any_count_mismatch"
            ].sum()
        )
    )

    print(
        "Missing from summary:",
        int(
            (
                employee_report["record_status"]
                == "MISSING_FROM_SUMMARY"
            ).sum()
        )
    )

    print(
        "Missing from raw:",
        int(
            (
                employee_report["record_status"]
                == "MISSING_FROM_RAW"
            ).sum()
        )
    )


# ============================================================
# 9. FINAL STATUS
# ============================================================

print("\n" + "=" * 75)

if (
    report["record_status"].eq(
        "MATCHED_EMPLOYEE_DAY"
    ).all()
    and not report["any_count_mismatch"].any()
):

    print("PASS: ALL EMPLOYEE-DAY COUNTS MATCH")

else:

    print("FAIL: EMPLOYEE-DAY DISCREPANCIES FOUND")

print("=" * 75)

print("\nReport saved to:")
print(report_path)

print("\nAUDIT COMPLETED")