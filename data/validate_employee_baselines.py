
import pandas as pd
import numpy as np
from pathlib import Path
import time


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(r"D:\Programs\AEGIS 2\data")

INPUT_FILE = BASE_DIR / "employee_baselines.csv"

ROLLING_WINDOW = "30D"

MIN_ROLLING_OBSERVATIONS = 5
MIN_LONG_TERM_OBSERVATIONS = 20

CHUNK_SIZE = 100_000
SAMPLE_SIZE = 10

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
# INDEPENDENT BASELINE RECOMPUTATION
# ==========================================================

def independently_recalculate(group, feature):

    active = group.loc[
        group[feature] > 0,
        ["date", feature]
    ].copy()

    if active.empty:
        return None

    active = active.sort_values("date")
    active = active.set_index("date")

    values = active[feature].astype(float)

    rolling = values.rolling(
        "30D",
        closed="left",
        min_periods=1
    )

    rolling_mean = rolling.mean()
    rolling_std = rolling.std()
    rolling_median = rolling.median()
    rolling_count = rolling.count()

    previous = values.shift(1)

    expanding = previous.expanding(min_periods=1)

    long_mean = expanding.mean()

    long_std = previous.expanding(
        min_periods=2
    ).std()

    long_median = expanding.median()
    long_count = expanding.count()

    rolling_z = np.full(len(values), np.nan)
    long_z = np.full(len(values), np.nan)

    rolling_mask = (
        (rolling_count.to_numpy() >= MIN_ROLLING_OBSERVATIONS)
        & np.isfinite(rolling_std.to_numpy())
        & (rolling_std.to_numpy() > 0)
        & np.isfinite(rolling_mean.to_numpy())
    )

    long_mask = (
        (long_count.to_numpy() >= MIN_LONG_TERM_OBSERVATIONS)
        & np.isfinite(long_std.to_numpy())
        & (long_std.to_numpy() > 0)
        & np.isfinite(long_mean.to_numpy())
    )

    np.divide(
        values.to_numpy() - rolling_mean.to_numpy(),
        rolling_std.to_numpy(),
        out=rolling_z,
        where=rolling_mask
    )

    np.divide(
        values.to_numpy() - long_mean.to_numpy(),
        long_std.to_numpy(),
        out=long_z,
        where=long_mask
    )

    return pd.DataFrame(
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
        index=active.index
    )


# ==========================================================
# NUMERICAL COMPARISON
# ==========================================================

def compare_values(actual, expected):

    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)

    return np.allclose(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True
    )


# ==========================================================
# MAIN VALIDATION
# ==========================================================

def main():

    start_time = time.time()

    print("=" * 65)
    print("AEGIS - EMPLOYEE BASELINE VALIDATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {INPUT_FILE}"
        )

    # ------------------------------------------------------
    # PREPARE REQUIRED COLUMNS
    # ------------------------------------------------------

    required_columns = ["user", "date"] + FEATURES

    baseline_suffixes = [
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

    baseline_columns = [
        f"{feature}_{suffix}"
        for feature in FEATURES
        for suffix in baseline_suffixes
    ]

    use_columns = required_columns + baseline_columns

    print("\nSelecting sample employees...")

    user_column = pd.read_csv(
        INPUT_FILE,
        usecols=["user"]
    )["user"]
    
    expected_rows = len(user_column)
    
    users = user_column.dropna().drop_duplicates()
    
    print(f"Expected dataset rows: {expected_rows:,}")

    sample_users = set(
        users.sample(
            n=min(SAMPLE_SIZE, len(users)),
            random_state=42
        )
    )

    print(f"Sample employees: {len(sample_users)}")

    # ------------------------------------------------------
    # FULL DATASET STRUCTURAL VALIDATION
    # ------------------------------------------------------

    print("\nChecking the complete dataset...")

    total_rows = 0
    missing_user_ids = 0
    negative_features = 0
    infinite_values = 0
    missing_feature_values = 0
    invalid_zscores = 0
    inactive_rows_with_baselines = 0

    rolling_ready = {feature: 0 for feature in FEATURES}
    long_ready = {feature: 0 for feature in FEATURES}

    sample_parts = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=use_columns,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        total_rows += len(chunk)
        missing_user_ids += int(chunk["user"].isna().sum())
        if chunk_number % 5 == 0:
            print(
                f"Checked {total_rows:,} rows..."
            )

        # Convert dates.
        chunk["date"] = pd.to_datetime(
            chunk["date"],
            errors="coerce"
        )

        if chunk["date"].isna().any():
            raise ValueError("Invalid dates found.")

        for feature in FEATURES:

            values = chunk[feature].to_numpy(dtype=float)
            missing_feature_values += int(
                np.isnan(values).sum()
            )
            if np.any(values < 0):
                negative_features += int(
                    np.sum(values < 0)
                )

            prefix = feature + "_"

            rolling_count = chunk[
                prefix + "rolling_count"
            ].to_numpy(dtype=float)

            long_count = chunk[
                prefix + "long_count"
            ].to_numpy(dtype=float)

            rolling_std = chunk[
                prefix + "rolling_std"
            ].to_numpy(dtype=float)

            long_std = chunk[
                prefix + "long_std"
            ].to_numpy(dtype=float)

            rolling_mean = chunk[
                prefix + "rolling_mean"
            ].to_numpy(dtype=float)

            long_mean = chunk[
                prefix + "long_mean"
            ].to_numpy(dtype=float)

            rolling_z = chunk[
                prefix + "rolling_zscore"
            ].to_numpy(dtype=float)

            long_z = chunk[
                prefix + "long_zscore"
            ].to_numpy(dtype=float)

            rolling_ready[feature] += int(
                np.sum(
                    rolling_count >= MIN_ROLLING_OBSERVATIONS
                )
            )

            long_ready[feature] += int(
                np.sum(
                    long_count >= MIN_LONG_TERM_OBSERVATIONS
                )
            )

            infinite_values += int(
                np.isinf(rolling_z).sum()
                + np.isinf(long_z).sum()
            )

            invalid_zscores += int(
                np.sum(
                    np.isfinite(rolling_z)
                    & (
                        (rolling_std <= 0)
                        | (rolling_count < MIN_ROLLING_OBSERVATIONS)
                        | ~np.isfinite(rolling_mean)
                    )
                )
            )

            invalid_zscores += int(
                np.sum(
                    np.isfinite(long_z)
                    & (
                        (long_std <= 0)
                        | (long_count < MIN_LONG_TERM_OBSERVATIONS)
                        | ~np.isfinite(long_mean)
                    )
                )
            )

            # Inactive observations must not receive
            # baselines under the current implementation.
            inactive = values <= 0

            related_columns = [
                prefix + suffix
                for suffix in baseline_suffixes
            ]

            inactive_baselines = chunk.loc[
                inactive,
                related_columns
            ]

            inactive_rows_with_baselines += int(
                inactive_baselines.notna().any(axis=1).sum()
            )

        # Retain sample employees for independent recalculation.
        sample_part = chunk[
            chunk["user"].isin(sample_users)
        ].copy()

        if not sample_part.empty:
            sample_parts.append(sample_part)

    # ------------------------------------------------------
    # STRUCTURAL VALIDATION RESULTS
    # ------------------------------------------------------

    print("\n" + "=" * 65)
    print("STRUCTURAL VALIDATION RESULTS")
    print("=" * 65)

    print(f"Total rows checked: {total_rows:,}")
    print(f"Negative feature values: {negative_features:,}")
    print(f"Missing employee IDs: {missing_user_ids:,}")
    print(f"Missing feature values: {missing_feature_values:,}")
    print(f"Infinite z-scores: {infinite_values:,}")
    print(f"Invalid finite z-scores: {invalid_zscores:,}")
    print(
        "Inactive rows with assigned baselines: "
        f"{inactive_rows_with_baselines:,}"
    )

    print("\nFEATURE-LEVEL BASELINE READINESS")

    for feature in FEATURES:

        print(
            f"{feature:30} "
            f"Rolling: {rolling_ready[feature]:,} | "
            f"Long-term: {long_ready[feature]:,}"
        )

    # ------------------------------------------------------
    # INDEPENDENT MATHEMATICAL VALIDATION
    # ------------------------------------------------------

    print("\n" + "=" * 65)
    print("INDEPENDENT MATHEMATICAL VALIDATION")
    print("=" * 65)

    if not sample_parts:
        raise ValueError(
            "No sample employee records were collected. "
            "Independent validation cannot proceed."
        )
    
    sample_df = pd.concat(
        sample_parts,
        ignore_index=True
    )

    sample_df = sample_df.sort_values(
        ["user", "date"]
    )

    checked_comparisons = 0
    failed_comparisons = []

    suffixes = [
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

    for user, group in sample_df.groupby("user"):

        group = group.sort_values("date")

        for feature in FEATURES:

            expected = independently_recalculate(
                group,
                feature
            )

            if expected is None:
                continue

            for suffix in suffixes:

                column = f"{feature}_{suffix}"

                actual = (
                    group.set_index("date")
                    .loc[expected.index, column]
                    .to_numpy(dtype=float)
                )

                expected_values = expected[
                    suffix
                ].to_numpy(dtype=float)

                checked_comparisons += 1

                if not compare_values(
                    actual,
                    expected_values
                ):

                    failed_comparisons.append(
                        f"User {user}, feature {feature}, "
                        f"column {suffix}"
                    )

    print(
        f"Sample employees independently checked: "
        f"{len(sample_users)}"
    )

    print(
        f"Numerical comparisons: {checked_comparisons:,}"
    )

    print(
        f"Failed comparisons: {len(failed_comparisons):,}"
    )

    for failure in failed_comparisons[:20]:
        print("FAILED:", failure)

    # ------------------------------------------------------
    # FINAL VERDICT
    # ------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL VALIDATION VERDICT")
    print("=" * 65)

    checks = {
        "No missing employee IDs": missing_user_ids == 0,
        "No missing feature values": missing_feature_values == 0,
        "All rows processed": total_rows == expected_rows,
        "No negative feature values": negative_features == 0,
        "No infinite z-scores": infinite_values == 0,
        "No invalid finite z-scores": invalid_zscores == 0,
        "No baselines on inactive rows": inactive_rows_with_baselines == 0,
        "Independent calculations match": (
            checked_comparisons > 0
            and len(failed_comparisons) == 0
)
    }

    for name, passed in checks.items():

        print(
            f"{'[PASS]' if passed else '[FAIL]'} {name}"
        )

    if all(checks.values()):

        print("\nBASELINE VALIDATION PASSED")

    else:

        print("\nBASELINE VALIDATION NEEDS INVESTIGATION")

    elapsed = time.time() - start_time

    print(
        f"\nExecution time: {elapsed / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()
