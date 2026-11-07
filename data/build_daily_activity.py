import pandas as pd
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path(__file__).resolve().parent

LOGON_FILE = DATA_DIR / "logon.csv"
FILE_FILE = DATA_DIR / "file.csv"
DEVICE_FILE = DATA_DIR / "device.csv"

OUTPUT_FILE = DATA_DIR / "daily_activity_summary.csv"

CHUNK_SIZE = 200_000


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def read_header(file_path):
    """Read CSV column names without loading the complete file."""
    return pd.read_csv(file_path, nrows=0).columns.tolist()


def validate_columns(file_path, required_columns):
    columns = read_header(file_path)

    missing = set(required_columns) - set(columns)

    if missing:
        raise ValueError(
            f"{file_path.name} is missing required columns: {missing}\n"
            f"Available columns: {columns}"
        )


def normalize_boolean(series):
    """Convert common CSV boolean representations into True/False."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def combine_daily(partials):
    """Combine partial daily aggregations from multiple chunks."""
    if not partials:
        return pd.DataFrame(columns=["user", "date"])

    combined = pd.concat(partials, ignore_index=True)

    numeric_columns = [
        column for column in combined.columns
        if column not in ["user", "date"]
    ]

    return (
        combined.groupby(["user", "date"], as_index=False)[numeric_columns]
        .sum()
    )


# =========================================================
# 1. LOGON ACTIVITY
# =========================================================

def process_logon():

    print("\n[1/3] Processing logon.csv...")

    validate_columns(
        LOGON_FILE,
        ["date", "user", "pc", "activity"]
    )

    partials = []
    pc_records = []

    for chunk in pd.read_csv(
        LOGON_FILE,
        usecols=["date", "user", "pc", "activity"],
        chunksize=CHUNK_SIZE
    ):

        chunk["date"] = pd.to_datetime(
            chunk["date"], errors="coerce"
        ).dt.date

        chunk = chunk.dropna(subset=["date", "user"])

        activity = chunk["activity"].astype(str).str.strip().str.lower()

        chunk["logon_events"] = (activity == "logon").astype(int)
        chunk["logoff_events"] = (activity == "logoff").astype(int)

        daily = chunk.groupby(
            ["user", "date"], as_index=False
        ).agg(
            logon_events=("logon_events", "sum"),
            logoff_events=("logoff_events", "sum")
        )

        partials.append(daily)

        pc_records.append(
            chunk[["user", "date", "pc"]].dropna().drop_duplicates()
        )

    result = combine_daily(partials)

    print(f"Logon daily records: {len(result):,}")

    return result, pc_records


# =========================================================
# 2. FILE ACTIVITY
# =========================================================

def process_file():

    print("\n[2/3] Processing file.csv in chunks...")

    validate_columns(
        FILE_FILE,
        [
            "date", "user", "pc", "activity",
            "to_removable_media", "from_removable_media"
        ]
    )

    partials = []
    pc_records = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            FILE_FILE,
            usecols=[
                "date", "user", "pc", "activity",
                "to_removable_media", "from_removable_media"
            ],
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        chunk["date"] = pd.to_datetime(
            chunk["date"], errors="coerce"
        ).dt.date

        chunk = chunk.dropna(subset=["date", "user"])

        activity = chunk["activity"].astype(str).str.strip().str.lower()

        chunk["file_events"] = 1
        chunk["file_copies"] = (activity == "file copy").astype(int)
        chunk["file_opens"] = (activity == "file open").astype(int)
        chunk["file_deletes"] = (activity == "file delete").astype(int)
        chunk["file_writes"] = (activity == "file write").astype(int)

        to_removable = normalize_boolean(chunk["to_removable_media"])
        from_removable = normalize_boolean(chunk["from_removable_media"])

        chunk["to_removable_events"] = to_removable.astype(int)
        chunk["from_removable_events"] = from_removable.astype(int)

        chunk["removable_media_events"] = (
            to_removable | from_removable
        ).astype(int)

        daily = chunk.groupby(
            ["user", "date"], as_index=False
        ).agg(
            file_events=("file_events", "sum"),
            file_copies=("file_copies", "sum"),
            file_opens=("file_opens", "sum"),
            file_deletes=("file_deletes", "sum"),
            file_writes=("file_writes", "sum"),
            to_removable_events=("to_removable_events", "sum"),
            from_removable_events=("from_removable_events", "sum"),
            removable_media_events=("removable_media_events", "sum")
        )

        partials.append(daily)

        pc_records.append(
            chunk[["user", "date", "pc"]].dropna().drop_duplicates()
        )

        print(f"  Processed chunk {chunk_number}")

    result = combine_daily(partials)

    print(f"File daily records: {len(result):,}")

    return result, pc_records


# =========================================================
# 3. DEVICE ACTIVITY
# =========================================================

def process_device():

    print("\n[3/3] Processing device.csv...")

    validate_columns(
        DEVICE_FILE,
        ["date", "user", "pc", "activity"]
    )

    partials = []
    pc_records = []

    for chunk in pd.read_csv(
        DEVICE_FILE,
        usecols=["date", "user", "pc", "activity"],
        chunksize=CHUNK_SIZE
    ):

        chunk["date"] = pd.to_datetime(
            chunk["date"], errors="coerce"
        ).dt.date

        chunk = chunk.dropna(subset=["date", "user"])

        activity = chunk["activity"].astype(str).str.strip().str.lower()

        chunk["device_events"] = 1
        chunk["device_connects"] = (activity == "connect").astype(int)
        chunk["device_disconnects"] = (activity == "disconnect").astype(int)

        daily = chunk.groupby(
            ["user", "date"], as_index=False
        ).agg(
            device_events=("device_events", "sum"),
            device_connects=("device_connects", "sum"),
            device_disconnects=("device_disconnects", "sum")
        )

        partials.append(daily)

        pc_records.append(
            chunk[["user", "date", "pc"]].dropna().drop_duplicates()
        )

    result = combine_daily(partials)

    print(f"Device daily records: {len(result):,}")

    return result, pc_records


# =========================================================
# 4. BUILD UNIFIED DAILY DATASET
# =========================================================

def main():

    print("=" * 60)
    print("AEGIS - DAILY ACTIVITY DATASET BUILDER")
    print("=" * 60)

    logon, logon_pcs = process_logon()
    file_data, file_pcs = process_file()
    device, device_pcs = process_device()

    print("\nCombining telemetry sources...")

    # Full outer join: retain activity from every source.
# Track source availability before merging.
    logon["logon_day_covered"] = True
    file_data["file_day_covered"] = True
    device["device_day_covered"] = True

    # Full outer join: retain activity from every source.
    daily = logon.merge(
        file_data,
        on=["user", "date"],
        how="outer"
    )

    daily = daily.merge(
        device,
        on=["user", "date"],
        how="outer"
    )

    # Missing source records indicate unavailable telemetry.
    coverage_columns = [
        "logon_day_covered",
        "file_day_covered",
        "device_day_covered"
    ]
    
    for column in coverage_columns:
        daily[column] = daily[column].fillna(False).astype(bool)
    
    # Identify whether each employee has ever appeared
    # in each telemetry source.
    logon_users = set(logon["user"])
    file_users = set(file_data["user"])
    device_users = set(device["user"])

    daily["logon_coverage"] = daily["user"].isin(logon_users)
    daily["file_coverage"] = daily["user"].isin(file_users)
    daily["device_coverage"] = daily["user"].isin(device_users)

    # Missing daily counts are represented as zero.
    # Coverage columns preserve the distinction between
    # absent telemetry and a day with no recorded events.
    count_columns = [
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
        "device_disconnects"
    ]

    for column in count_columns:
        daily[column] = daily[column].fillna(0).astype("int64")

    # Calculate distinct PCs across all three sources.
    print("Calculating unique PCs...")

    pc_frames = logon_pcs + file_pcs + device_pcs

    pc_data = pd.concat(pc_frames, ignore_index=True)

    pc_data["pc"] = pc_data["pc"].astype(str).str.strip()

    pc_data = pc_data[
        pc_data["pc"].ne("") &
        pc_data["pc"].str.lower().ne("nan")
    ].drop_duplicates(["user", "date", "pc"])

    unique_pcs = pc_data.groupby(
        ["user", "date"]
    )["pc"].nunique().reset_index(name="unique_pcs")

    daily = daily.merge(
        unique_pcs,
        on=["user", "date"],
        how="left"
    )

    daily["unique_pcs"] = daily["unique_pcs"].fillna(0).astype(int)

    # Add total recorded events across the three sources.
    daily["total_events"] = (
        daily["logon_events"]
        + daily["logoff_events"]
        + daily["file_events"]
        + daily["device_events"]
    )

    # Final ordering.
    daily["date"] = pd.to_datetime(daily["date"])

    daily = daily.sort_values(
        ["user", "date"]
    ).reset_index(drop=True)

    # Save.
    daily.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("DATASET BUILD COMPLETED")
    print("=" * 60)

    print(f"Output: {OUTPUT_FILE}")
    print(f"Total daily records: {len(daily):,}")
    print(f"Unique employees: {daily['user'].nunique():,}")
    print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")

    print("\nCoverage:")
    print(f"Logon employees: {daily.loc[daily['logon_coverage'], 'user'].nunique():,}")
    print(f"File employees: {daily.loc[daily['file_coverage'], 'user'].nunique():,}")
    print(f"Device employees: {daily.loc[daily['device_coverage'], 'user'].nunique():,}")

    print("\nFinal columns:")
    print(daily.columns.tolist())

    print("\nSample records:")
    print(daily.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
    