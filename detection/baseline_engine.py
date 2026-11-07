

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DEFAULT_WINDOW_DAYS = 30
DEFAULT_MIN_OBSERVATIONS = 5

REQUIRED_COLUMNS = [
    "user",
    "calendar_date",
    "logon_count",
    "first_login_minutes",
    "computers_used",
    "has_logon_activity"
]


# ---------------------------------------------------------
# 1. PREPARE DAILY DATA
# ---------------------------------------------------------

def prepare_daily_data(daily_features):
    """
    Validate and prepare employee-day feature data.
    """

    if not isinstance(daily_features, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in daily_features.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = daily_features.copy()

    # Normalize employee IDs.
    df["user"] = df["user"].astype("string").str.strip()

    # Parse dates.
    df["calendar_date"] = pd.to_datetime(
        df["calendar_date"],
        errors="coerce"
    ).dt.normalize()

    # Convert login counts.
    df["logon_count"] = pd.to_numeric(
        df["logon_count"],
        errors="coerce"
    )

    # Convert login times.
    df["first_login_minutes"] = pd.to_numeric(
        df["first_login_minutes"],
        errors="coerce"
    )
    if "last_login_minutes" not in df.columns:
        df["last_login_minutes"] = pd.NA
    df["last_login_minutes"] = pd.to_numeric(
        df["last_login_minutes"],
        errors="coerce"
    )
    # Normalize Boolean activity and telemetry coverage flags.
    def normalize_boolean_column(series):
        return (
            series.astype("string")
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False
            })
        )
    
    
    df["has_logon_activity"] = normalize_boolean_column(
        df["has_logon_activity"]
    )
    
    for column in [
        "logon_day_covered",
        "file_day_covered",
        "device_day_covered"
    ]:
        if column in df.columns:
            df[column] = normalize_boolean_column(df[column])

    # Normalize computer information.
    df["computers_used"] = (
        df["computers_used"]
        .astype("string")
        .str.strip()
    )

    # Remove records without employee IDs or valid dates.
    df = df.dropna(
        subset=["user", "calendar_date"]
    )

    # Validate employee-day uniqueness.
    duplicate_count = df.duplicated(
        subset=["user", "calendar_date"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate employee-day records."
        )

    return df


# ---------------------------------------------------------
# 2. LOGIN FREQUENCY BASELINE
# ---------------------------------------------------------

def calculate_login_frequency_baseline(
    history,
    min_observations=DEFAULT_MIN_OBSERVATIONS
):
    """
    Calculate login frequency statistics.

    Only employee-days with recorded login activity are used.

    This is a conditional login-frequency baseline,
    NOT an average across all calendar days.
    """

    login_days = history[
        (history["has_logon_activity"] == True)
        & (history["logon_count"] > 0)
    ]["logon_count"].dropna()

    observation_count = len(login_days)

    if observation_count < min_observations:
        return {
            "baseline_available": False,
            "observation_count": observation_count,
            "reason": "Insufficient login observations"
        }

    return {
        "baseline_available": True,
        "observation_count": observation_count,
        "mean": float(login_days.mean()),
        "median": float(login_days.median()),
        "std": float(login_days.std(ddof=0)),
        "minimum": int(login_days.min()),
        "maximum": int(login_days.max()),
        "p95": float(login_days.quantile(0.95))
    }


# ---------------------------------------------------------
# 3. LOGIN TIMING BASELINE
# ---------------------------------------------------------

def calculate_login_time_baseline(
    history,
    min_observations=DEFAULT_MIN_OBSERVATIONS
):
    """
    Calculate employee-specific login timing statistics.

    Uses circular statistics to handle the midnight boundary.

    Example:
        23:50 and 00:10 should have a mean near midnight,
        not noon.
    """

    login_times = history.loc[
        (history["has_logon_activity"] == True)
        & (history["first_login_minutes"].notna()),
        "first_login_minutes"
    ]

    # Valid times must fall within one calendar day.
    login_times = login_times[
        login_times.between(0, 1439)
    ]

    observation_count = len(login_times)

    if observation_count < min_observations:
        return {
            "baseline_available": False,
            "observation_count": observation_count,
            "reason": "Insufficient login-time observations"
        }

    # Convert minutes into angles.
    angles = (
        2 * np.pi * login_times.to_numpy() / 1440
    )

    sin_mean = np.sin(angles).mean()
    cos_mean = np.cos(angles).mean()

    # Circular mean.
    mean_angle = np.arctan2(
        sin_mean,
        cos_mean
    )

    circular_mean_minutes = (
        (mean_angle % (2 * np.pi))
        * 1440 / (2 * np.pi)
    )

    # Resultant length measures concentration.
    resultant_length = np.sqrt(
        sin_mean ** 2 + cos_mean ** 2
    )

    return {
        "baseline_available": True,
        "observation_count": observation_count,
        "circular_mean_minutes": float(
            circular_mean_minutes
        ),
        "resultant_length": float(
            resultant_length
        )
    }
def calculate_last_login_time_baseline(
    history,
    min_observations=DEFAULT_MIN_OBSERVATIONS
):
    """Calculate the circular baseline for employees' last login times."""

    login_times = history.loc[
        (history["has_logon_activity"] == True)
        & (history["last_login_minutes"].notna()),
        "last_login_minutes"
    ]

    login_times = login_times[
        login_times.between(0, 1439)
    ]

    observation_count = len(login_times)

    if observation_count < min_observations:
        return {
            "baseline_available": False,
            "observation_count": observation_count,
            "reason": "Insufficient last-login observations"
        }

    angles = 2 * np.pi * login_times.to_numpy() / 1440

    sin_mean = np.sin(angles).mean()
    cos_mean = np.cos(angles).mean()

    mean_angle = np.arctan2(sin_mean, cos_mean)

    circular_mean_minutes = (
        (mean_angle % (2 * np.pi)) * 1440 / (2 * np.pi)
    )

    resultant_length = np.sqrt(
        sin_mean ** 2 + cos_mean ** 2
    )

    return {
        "baseline_available": True,
        "observation_count": observation_count,
        "circular_mean_minutes": float(circular_mean_minutes),
        "resultant_length": float(resultant_length)
    }

# ---------------------------------------------------------
# 4. COMPUTER USAGE BASELINE
# ---------------------------------------------------------

def calculate_computer_baseline(
    history,
    min_observations=DEFAULT_MIN_OBSERVATIONS,
    frequent_usage_threshold=20.0,
    min_frequent_days=2
):
    

    computer_usage = {}

    valid_computer_days = 0

    # Process each historical employee-day.
    for _, row in history.iterrows():
        if not bool(row["has_logon_activity"]):
            continue
        computer_list = row["computers_used"]
        calendar_date = row["calendar_date"]

        if pd.isna(computer_list):
            continue

        computers = {
            computer.strip()
            for computer in str(computer_list).split("|")
            if computer.strip()
        }

        if not computers:
            continue

        valid_computer_days += 1

        # Count each computer at most once per employee-day.
        for computer in computers:

            if computer not in computer_usage:

                computer_usage[computer] = {
                    "days_used": 0,
                    "last_seen": calendar_date
                }

            computer_usage[computer]["days_used"] += 1

            if calendar_date > computer_usage[computer]["last_seen"]:

                computer_usage[computer]["last_seen"] = calendar_date

    # Insufficient historical observations.
    if valid_computer_days < min_observations:

        return {
            "baseline_available": False,
            "observation_count": valid_computer_days,
            "known_computers": [],
            "known_computer_count": 0,
            "frequent_computers": [],
            "occasional_computers": [],
            "reason": "Insufficient computer-usage observations"
        }

    # Build individual computer profiles.
    computer_profiles = []

    for computer, details in computer_usage.items():

        days_used = details["days_used"]

        usage_percentage = (
            days_used / valid_computer_days
        ) * 100

        if (
            usage_percentage >= frequent_usage_threshold
            and days_used >= min_frequent_days
        ):

            classification = "REGULAR"

        else:

            classification = "OCCASIONAL"

        computer_profiles.append({

            "computer_id": computer,

            "days_used": days_used,

            "usage_percentage": round(
                usage_percentage, 2
            ),

            "last_seen": details["last_seen"].strftime(
                "%Y-%m-%d"
            ),

            "classification": classification

        })

    # Sort by frequency of usage.
    computer_profiles.sort(
        key=lambda x: (
            x["days_used"],
            x["computer_id"]
        ),
        reverse=True
    )

    frequent_computers = [
        profile["computer_id"]
        for profile in computer_profiles
        if profile["classification"] == "REGULAR"
    ]

    occasional_computers = [
        profile["computer_id"]
        for profile in computer_profiles
        if profile["classification"] == "OCCASIONAL"
    ]

    known_computers = [
        profile["computer_id"]
        for profile in computer_profiles
    ]

    return {

        "baseline_available": True,

        "observation_count": valid_computer_days,

        "known_computers": known_computers,

        "known_computer_count": len(known_computers),

        "frequent_computers": frequent_computers,

        "frequent_computer_count": len(frequent_computers),

        "occasional_computers": occasional_computers,

        "occasional_computer_count": len(occasional_computers),

        "computer_profiles": computer_profiles,

        "frequent_usage_threshold": frequent_usage_threshold,

        "min_frequent_days": min_frequent_days

    }

# ---------------------------------------------------------
# 5. BUILD EMPLOYEE-SPECIFIC BASELINE
# ---------------------------------------------------------

def build_employee_baseline(
    daily_features,
    user,
    target_date,
    window_days=DEFAULT_WINDOW_DAYS,
    min_observations=DEFAULT_MIN_OBSERVATIONS
):
    

    if window_days < 1:
        raise ValueError("window_days must be at least 1.")

    if min_observations < 1:
        raise ValueError("min_observations must be at least 1.")

    if not user:
        raise ValueError("Employee ID cannot be empty.")

    # Prepare input.
    df = prepare_daily_data(daily_features)

    target_date = pd.Timestamp(
        target_date
    ).normalize()

    window_start = (
        target_date - pd.Timedelta(days=window_days)
    )

    # Select only the specified employee's historical records.
    history = df[
        (df["user"] == user)
        & (df["calendar_date"] >= window_start)
        & (df["calendar_date"] < target_date)
    ].copy()

    # Sort historical observations.
    history = history.sort_values(
        "calendar_date"
    )

    # Independent baseline calculations.
    frequency_baseline = calculate_login_frequency_baseline(
        history,
        min_observations
    )

    timing_baseline = calculate_login_time_baseline(
        history,
        min_observations
    )

    last_login_timing_baseline = calculate_last_login_time_baseline(
        history,
        min_observations
    )
    computer_baseline = calculate_computer_baseline(
        history,
        min_observations
    )

    # Final result.
    result = {
        "user": user,
        "target_date": target_date.strftime("%Y-%m-%d"),

        "window_days": window_days,

        "window_start": window_start.strftime("%Y-%m-%d"),

        "window_end_exclusive": target_date.strftime("%Y-%m-%d"),

        "historical_employee_days": len(history),

        "login_frequency": frequency_baseline,

        "login_timing": timing_baseline,

        "last_login_timing": last_login_timing_baseline,

        "computer_usage": computer_baseline
    }

    return result


# ---------------------------------------------------------
# 6. MAIN EXECUTION
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("AEGIS - BEHAVIOURAL BASELINE ENGINE")
    print("=" * 60)

    input_path = "data/daily_activity_features.csv"

    print("\nLoading employee-day features...")

    df = pd.read_csv(input_path)

    print("Records loaded:", len(df))

    # Example employee and target date.
    test_user = "AAB0162"
    test_date = "2010-02-01"

    print("\nBuilding employee baseline...")

    baseline = build_employee_baseline(
        daily_features=df,
        user=test_user,
        target_date=test_date,
        window_days=30,
        min_observations=5
    )

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)

    print("\nEmployee:", baseline["user"])
    print("Target date:", baseline["target_date"])

    print(
        "Historical window:",
        baseline["window_start"],
        "to",
        baseline["window_end_exclusive"],
        "(exclusive)"
    )

    print(
        "\nHistorical employee-days:",
        baseline["historical_employee_days"]
    )

    print("\nLOGIN FREQUENCY")
    print("-" * 40)

    for key, value in baseline["login_frequency"].items():
        print(f"{key}: {value}")

    print("\nLOGIN TIMING")
    print("-" * 40)

    for key, value in baseline["login_timing"].items():
        print(f"{key}: {value}")

    print("\nCOMPUTER USAGE")
    print("-" * 40)

    for key, value in baseline["computer_usage"].items():

        if key == "known_computers":
            print(f"{key}: {value[:20]}")

        else:
            print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("BASELINE CALCULATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()