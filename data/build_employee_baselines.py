import pandas as pd
import numpy as np
from pathlib import Path
import time


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(r"D:\Programs\AEGIS 2\data")

INPUT_FILE = BASE_DIR / "daily_activity_summary.csv"
OUTPUT_FILE = BASE_DIR / "employee_baselines.csv"

ROLLING_WINDOW = "30D"

MIN_ROLLING_OBSERVATIONS = 5
MIN_LONG_TERM_OBSERVATIONS = 20


# ==========================================================
# FEATURES USED FOR BEHAVIOURAL BASELINES
# ==========================================================

FEATURES = [
    "logon_events",
    "logoff_events",
    "file_events",
    "file_copies",
    "file_opens",
    "file_deletes",
    "file_writes",
    "removable_media_events",
    "to_removable_events",
    "from_removable_events",
    "device_events",
    "device_connects",
    "device_disconnects"
]


# ==========================================================
# CALCULATE BASELINES FOR ONE EMPLOYEE AND ONE FEATURE
# ==========================================================

def calculate_baselines(group, feature):

    # Only positive recorded observations are used.
    active = group.loc[
        group[feature] > 0,
        ["date", feature]
    ].copy()

    result = pd.DataFrame(index=group.index)

    columns = [
        "rolling_mean",
        "rolling_std",
        "rolling_median",
        "rolling_count",
        "long_mean",
        "long_std",
        "long_median",
        "long_count",
        "rolling_zscore",
        "long_zscore"
    ]

    # Initialize all output columns.
    for column in columns:
        result[column] = np.nan

    # No positive observations: return unavailable baselines.
    if active.empty:
        return result

    # Preserve original dataframe indexes for alignment.
    original_indices = active.index.to_numpy()

    # Sort observations chronologically.
    active = active.sort_values("date")

    active = active.set_index("date")

    values = active[feature].astype(float)

    current_values = values.to_numpy()

    # ------------------------------------------------------
    # ROLLING 30-DAY BASELINE
    # ------------------------------------------------------

    # closed="left" excludes the current observation.
    rolling = values.rolling(
        ROLLING_WINDOW,
        closed="left",
        min_periods=1
    )

    rolling_mean = rolling.mean()
    rolling_std = rolling.std()
    rolling_median = rolling.median()
    rolling_count = rolling.count()

    rolling_valid = (
        rolling_count >= MIN_ROLLING_OBSERVATIONS
    )

    # ------------------------------------------------------
    # LONG-TERM HISTORICAL BASELINE
    # ------------------------------------------------------

    # shift(1) excludes the current observation.
    previous = values.shift(1)

    expanding = previous.expanding(
        min_periods=1
    )

    long_mean = expanding.mean()

    long_std = previous.expanding(
        min_periods=2
    ).std()

    long_median = expanding.median()

    long_count = expanding.count()

    long_valid = (
        long_count >= MIN_LONG_TERM_OBSERVATIONS
    )

    # ------------------------------------------------------
    # SAFE Z-SCORE CALCULATION
    # ------------------------------------------------------

    rolling_mean_values = rolling_mean.to_numpy()
    rolling_std_values = rolling_std.to_numpy()

    long_mean_values = long_mean.to_numpy()
    long_std_values = long_std.to_numpy()

    rolling_z = np.full(
        len(current_values),
        np.nan
    )

    long_z = np.full(
        len(current_values),
        np.nan
    )

    # Only calculate when sufficient history exists
    # and standard deviation is finite and positive.

    rolling_mask = (
        rolling_valid.to_numpy()
        & np.isfinite(rolling_std_values)
        & (rolling_std_values > 0)
        & np.isfinite(rolling_mean_values)
    )

    long_mask = (
        long_valid.to_numpy()
        & np.isfinite(long_std_values)
        & (long_std_values > 0)
        & np.isfinite(long_mean_values)
    )

    np.divide(
        current_values - rolling_mean_values,
        rolling_std_values,
        out=rolling_z,
        where=rolling_mask
    )

    np.divide(
        current_values - long_mean_values,
        long_std_values,
        out=long_z,
        where=long_mask
    )

    # ------------------------------------------------------
    # STORE RESULTS
    # ------------------------------------------------------

    calculated = pd.DataFrame(
        {
            "rolling_mean": rolling_mean.to_numpy(),
            "rolling_std": rolling_std.to_numpy(),
            "rolling_median": rolling_median.to_numpy(),
            "rolling_count": rolling_count.to_numpy(),

            "long_mean": long_mean.to_numpy(),
            "long_std": long_std.to_numpy(),
            "long_median": long_median.to_numpy(),
            "long_count": long_count.to_numpy(),

            "rolling_zscore": rolling_z,
            "long_zscore": long_z
        },
        index=original_indices
    )

    result.loc[original_indices, columns] = calculated[columns]

    return result


# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main():

    start_time = time.time()

    print("=" * 65)
    print("AEGIS - EMPLOYEE BASELINE GENERATION")
    print("=" * 65)

    # ------------------------------------------------------
    # CHECK INPUT
    # ------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print("\nLoading daily activity summary...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded rows: {len(df):,}")
    print(f"Unique employees: {df['user'].nunique():,}")

    # ------------------------------------------------------
    # VALIDATE COLUMNS
    # ------------------------------------------------------

    required_columns = [
        "user",
        "date"
    ] + FEATURES

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ------------------------------------------------------
    # PREPARE DATA
    # ------------------------------------------------------

    print("\nPreparing data...")

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise"
    )

    if df.duplicated(["user", "date"]).any():
        raise ValueError(
            "Duplicate employee-date records detected."
        )

    df = df.sort_values(
        ["user", "date"]
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # GENERATE EMPLOYEE BASELINES
    # ------------------------------------------------------

    print("\nGenerating employee-specific baselines...")

    baseline_parts = []

    employee_groups = df.groupby(
        "user",
        sort=False
    )

    total_employees = employee_groups.ngroups

    for employee_number, (user, group) in enumerate(
        employee_groups,
        start=1
    ):

        employee_baselines = pd.DataFrame(
            index=group.index
        )

        for feature in FEATURES:

            feature_baselines = calculate_baselines(
                group,
                feature
            )

            feature_baselines = feature_baselines.add_prefix(
                f"{feature}_"
            )

            employee_baselines = employee_baselines.join(
                feature_baselines
            )

        baseline_parts.append(employee_baselines)

        if (
            employee_number % 100 == 0
            or employee_number == total_employees
        ):
            print(
                f"Processed {employee_number:,} "
                f"/ {total_employees:,} employees"
            )

    # ------------------------------------------------------
    # COMBINE BASELINES
    # ------------------------------------------------------

    print("\nCombining baseline results...")

    baseline_df = pd.concat(
        baseline_parts
    ).reindex(df.index)

    output_df = pd.concat(
        [df, baseline_df],
        axis=1
    )

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    print("\nValidating generated baselines...")

    baseline_columns = [
        column
        for column in baseline_df.columns
        if column.endswith("_zscore")
    ]

    rolling_columns = [
        column
        for column in baseline_df.columns
        if column.endswith("_rolling_count")
    ]

    long_columns = [
        column
        for column in baseline_df.columns
        if column.endswith("_long_count")
    ]

    print(f"Observed rows: {len(output_df):,}")

    if rolling_columns:
        rolling_ready = (
            baseline_df[rolling_columns]
            >= MIN_ROLLING_OBSERVATIONS
        ).any(axis=1).sum()

        print(
            f"Rows with at least one rolling baseline ready: "
            f"{rolling_ready:,}"
        )

    if long_columns:
        long_ready = (
            baseline_df[long_columns]
            >= MIN_LONG_TERM_OBSERVATIONS
        ).any(axis=1).sum()

        print(
            f"Rows with at least one long-term baseline ready: "
            f"{long_ready:,}"
        )

    print(
        f"Available z-score values: "
        f"{baseline_df[baseline_columns].notna().sum().sum():,}"
    )

    # Check for infinite z-scores.
    infinite_scores = np.isinf(
        baseline_df[baseline_columns].to_numpy(
            dtype=float
        )
    ).sum()

    print(f"Infinite z-scores: {infinite_scores:,}")

    if infinite_scores > 0:
        raise ValueError(
            "Infinite z-scores detected."
        )

    # ------------------------------------------------------
    # SAVE OUTPUT
    # ------------------------------------------------------

    print("\nSaving employee baselines...")

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print("BASELINE GENERATION COMPLETED")
    print("=" * 65)

    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total rows: {len(output_df):,}")
    print(f"Total columns: {len(output_df.columns):,}")
    print(f"Execution time: {elapsed / 60:.2f} minutes")


# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()