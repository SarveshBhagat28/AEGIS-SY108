
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "id",
    "date",
    "user",
    "pc",
    "activity",
}


def load_cert_logon_data(file_path):
    """
    Load and validate CERT logon.csv.

    Returns:
        DataFrame with normalized column names and timestamps.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Logon dataset not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    # Normalize column names.
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # Validate required columns.
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Keep only the required fields.
    df = df[list(REQUIRED_COLUMNS)].copy()

    # Normalize text fields.
    for column in ["id", "user", "pc", "activity"]:
        df[column] = df[column].astype("string").str.strip()

    # Parse timestamps.
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Identify invalid records.
    invalid_dates = df["date"].isna().sum()

    missing_values = df[
        ["user", "pc", "activity"]
    ].isna().sum().sum()

    # Remove records that cannot be used reliably.
    df = df.dropna(
        subset=["date", "user", "pc", "activity"]
    ).copy()

    # Normalize activity labels.
    df["activity"] = df["activity"].str.strip().str.lower()

    # Sort chronologically.
    df = df.sort_values(
        ["user", "date"]
    ).reset_index(drop=True)

    print("\n========== CERT LOGON DATA REPORT ==========")
    print(f"File: {file_path}")
    print(f"Valid records: {len(df):,}")
    print(f"Unique employees: {df['user'].nunique():,}")
    print(f"Unique computers: {df['pc'].nunique():,}")
    print(f"Invalid timestamps: {invalid_dates:,}")
    print(f"Missing required values: {missing_values:,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    print("\nActivity distribution:")
    print(df["activity"].value_counts())

    print("\nSample records:")
    print(df.head())

    return df
