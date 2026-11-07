from pathlib import Path
import pandas as pd
import sys

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

EVENTS_FILE = SCRIPT_DIR / "synthetic_logon_events.csv"
MANIFEST_FILE = SCRIPT_DIR / "synthetic_test_cases.csv"

OUTPUT_FILE = SCRIPT_DIR / "synthetic_test_results.csv"


# --------------------------------------------------
# LOAD INPUT FILES
# --------------------------------------------------

def load_test_data():

    print("\n========== AEGIS SYNTHETIC TEST RUNNER ==========\n")

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(f"Events file not found: {EVENTS_FILE}")

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Test manifest not found: {MANIFEST_FILE}")

    events = pd.read_csv(EVENTS_FILE)
    manifest = pd.read_csv(MANIFEST_FILE)

    print("EVENT DATA")
    print(f"Total events: {len(events):,}")
    print(f"Unique employees: {events['user'].nunique():,}")

    print("\nTEST MANIFEST")
    print(f"Total test cases: {len(manifest)}")

    return events, manifest


# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------

def validate_test_data(events, manifest):

    required_event_columns = {
        "id", "date", "user", "pc", "activity"
    }

    required_manifest_columns = {
        "test_id",
        "user",
        "description",
        "target_date",
        "test_feature",
        "expected_result",
        "telemetry_available"
    }

    errors = []

    missing_event_columns = required_event_columns - set(events.columns)
    missing_manifest_columns = required_manifest_columns - set(manifest.columns)

    if missing_event_columns:
        errors.append(f"Missing event columns: {missing_event_columns}")

    if missing_manifest_columns:
        errors.append(f"Missing manifest columns: {missing_manifest_columns}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return False

    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    manifest["target_date"] = pd.to_datetime(
        manifest["target_date"], errors="coerce"
    )

    if events["date"].isna().any():
        errors.append("Invalid or missing event timestamps detected.")

    if events["user"].isna().any():
        errors.append("Events contain missing employee IDs.")

    if events["id"].duplicated().any():
        errors.append("Duplicate event IDs detected.")

    if manifest["test_id"].duplicated().any():
        errors.append("Duplicate test IDs detected.")

    if manifest["user"].duplicated().any():
        errors.append("Multiple test cases use the same employee ID.")

    valid_results = {
        "NORMAL",
        "POTENTIAL_ANOMALY",
        "BASELINE_UNAVAILABLE",
        "UNKNOWN"
    }

    invalid_results = set(manifest["expected_result"]) - valid_results

    if invalid_results:
        errors.append(f"Invalid expected results: {invalid_results}")

    event_users = set(events["user"].unique())
    manifest_users = set(manifest["user"].unique())

    unknown_users = event_users - manifest_users

    if unknown_users:
        errors.append(f"Events belong to unknown test users: {unknown_users}")

    # Every telemetry-available case should have a target date.
    missing_target_dates = manifest[
        manifest["telemetry_available"] == True
    ]["target_date"].isna()

    if missing_target_dates.any():
        errors.append("Telemetry-available tests have missing target dates.")

    if errors:
        print("\nVALIDATION FAILED")

        for error in errors:
            print(f" - {error}")

        return False

    print("\nINPUT VALIDATION: PASSED")

    return True

# --------------------------------------------------
# DAILY FEATURE ENGINEERING
# --------------------------------------------------

def build_daily_features(events):

    print("\nBuilding daily employee features...")

    events = events.copy()

    events["date"] = pd.to_datetime(events["date"])

    events["calendar_date"] = events["date"].dt.normalize()

    # Separate login and logoff events
    events["is_logon"] = events["activity"].str.lower().eq("logon")
    events["is_logoff"] = events["activity"].str.lower().eq("logoff")

    # Convert timestamps into minutes since midnight
    events["login_minutes"] = (
        events["date"].dt.hour * 60
        + events["date"].dt.minute
    )

    # Define after-hours login:
    # Before 07:00 or at/after 19:00
    events["hour"] = events["date"].dt.hour

    events["is_after_hours"] = (
        events["is_logon"]
        & (
            (events["hour"] < 7)
            | (events["hour"] >= 19)
        )
    )

    # Keep actual login records for timestamp aggregation
    login_events = events[events["is_logon"]].copy()

    # Aggregate login features
    login_daily = (
        login_events.groupby(["user", "calendar_date"])
        .agg(
            logon_events=("activity", "size"),
            first_login_minutes=("login_minutes", "min"),
            last_login_minutes=("login_minutes", "max"),
            after_hours_logins=("is_after_hours", "sum"),
            unique_pcs=("pc", "nunique"),
            computers_used=("pc", lambda values: sorted(
                values.dropna().unique().tolist()
            ))
        )
        .reset_index()
    )

    # Aggregate logoff events separately
    logoff_daily = (
        events[events["is_logoff"]]
        .groupby(["user", "calendar_date"])
        .size()
        .reset_index(name="logoff_events")
    )

    # Combine daily login and logoff features
    daily = login_daily.merge(
        logoff_daily,
        on=["user", "calendar_date"],
        how="left"
    )

    daily["logoff_events"] = (
        daily["logoff_events"].fillna(0).astype(int)
    )

    # Convert minutes back to decimal hours for display/compatibility
    daily["first_login_hour"] = daily["first_login_minutes"] / 60
    daily["last_login_hour"] = daily["last_login_minutes"] / 60

    print(f"Daily feature rows: {len(daily):,}")
    print(f"Employees: {daily['user'].nunique():,}")

    return daily

# --------------------------------------------------
# EMPLOYEE BASELINE PREPARATION
# --------------------------------------------------

def prepare_employee_baseline(
    daily_features,
    user,
    target_date,
    feature="logon_events",
    window_days=30,
    minimum_observations=5
):

    target_date = pd.Timestamp(target_date).normalize()

    window_start = target_date - pd.Timedelta(days=window_days)

    # Select only the employee's historical activity.
    # The target date is strictly excluded.

    history = daily_features[
        (daily_features["user"] == user)
        & (daily_features["calendar_date"] >= window_start)
        & (daily_features["calendar_date"] < target_date)
    ].copy()

    active_history = history[
        (history["logon_events"] > 0)
        & (history[feature].notna())
    ]

    observation_count = len(active_history)

    baseline = {
        "user": user,
        "feature": feature,
        "target_date": target_date,
        "window_start": window_start,
        "observation_count": observation_count,
        "baseline_available": observation_count >= minimum_observations,
        "mean": None,
        "std": None,
        "median": None
    }

    if observation_count >= minimum_observations:
        baseline["available"] = True
        baseline["mean"] = active_history[feature].mean()
        baseline["std"] = active_history[feature].std()
        baseline["median"] = active_history[feature].median()
    else:
        baseline["available"] = False
        baseline["mean"] = None
        baseline["std"] = None
        baseline["median"] = None
    return baseline


# --------------------------------------------------
# SYNTHETIC FEATURE VALIDATION
# --------------------------------------------------

def validate_synthetic_features(daily_features, manifest):

    print("\n========== SYNTHETIC FEATURE VALIDATION ==========")

    checks = []

    for _, test in manifest.iterrows():

        test_id = test["test_id"]
        user = test["user"]
        target_date = test["target_date"]

        employee_data = daily_features[
            daily_features["user"] == user
        ]

        if pd.isna(target_date):
            checks.append((test_id, "PASS", "No target date required"))
            continue

        target_date = pd.Timestamp(target_date)

        target_data = employee_data[
            employee_data["calendar_date"] == target_date
        ]

        if not test["telemetry_available"]:
            checks.append((test_id, "PASS", "Telemetry unavailable"))
            continue

        if target_data.empty:
            checks.append((test_id, "FAIL", "Target date has no activity"))
            continue

        row = target_data.iloc[0]

        if test_id == "SYN03":
            passed = row["logon_events"] == 8
            detail = f"Target-day logins: {row['logon_events']} (expected 8)"

        elif test_id == "SYN05":
            window_start = target_date - pd.Timedelta(days=4)

            window = employee_data[
                employee_data["calendar_date"].between(
                    window_start, target_date
                )
            ]

            count = window["after_hours_logins"].sum()
            passed = count == 5
            detail = f"After-hours logins in 5-day window: {count} (expected 5)"

        elif test_id == "SYN08":
            target_computers = set(row["computers_used"])

            historical_features = daily_features[
                (daily_features["user"] == user)
                & (daily_features["calendar_date"] >= target_date - pd.Timedelta(days=30))
                & (daily_features["calendar_date"] < target_date)
            ]

            historical_computers = set(
                computer
                for computers in historical_features["computers_used"]
                for computer in computers
            )

            passed = (
                historical_computers == {"PC001"}
                and target_computers == {"PC999"}
                and len(historical_computers.intersection(target_computers)) == 0
            )

            detail = (
                f"Historical computers: {sorted(historical_computers)}, "
                f"Target computers: {sorted(target_computers)}"
            )

        elif test_id == "SYN09":
            passed = (
                row["logon_events"] == 3
                and row["after_hours_logins"] == 3
            )
            detail = (
                f"Logins: {row['logon_events']}, "
                f"after-hours logins: {row['after_hours_logins']}"
            )

        else:
            passed = True
            detail = "Target-day activity exists"

        checks.append((
            test_id,
            "PASS" if passed else "FAIL",
            detail
        ))

    for test_id, status, detail in checks:
        print(f"{status} | {test_id} | {detail}")

    failed = sum(status == "FAIL" for _, status, _ in checks)

    print(f"\nPassed: {len(checks) - failed}/{len(checks)}")
    print(f"Failed: {failed}/{len(checks)}")

    return failed == 0
# --------------------------------------------------
# BASELINE AVAILABILITY TEST
# --------------------------------------------------

def test_baseline_availability(daily_features, manifest):

    print("\n========== BASELINE AVAILABILITY TEST ==========")

    results = []

    for _, test in manifest.iterrows():

        if not test["telemetry_available"]:
            expected = "UNKNOWN"
            actual = "UNKNOWN"

            results.append({
                "test_id": test["test_id"],
                "observations": 0,
                "expected": expected,
                "actual": actual,
                "status": "PASS"
            })

            continue

        baseline = prepare_employee_baseline(
            daily_features=daily_features,
            user=test["user"],
            target_date=test["target_date"],
            feature="logon_events"
        )

        actual = (
            "AVAILABLE"
            if baseline["baseline_available"]
            else "UNAVAILABLE"
        )

        expected = (
            "UNAVAILABLE"
            if test["expected_result"] == "BASELINE_UNAVAILABLE"
            else "AVAILABLE"
        )

        status = "PASS" if actual == expected else "FAIL"

        results.append({
            "test_id": test["test_id"],
            "observations": baseline["observation_count"],
            "expected": expected,
            "actual": actual,
            "status": status
        })

    results_df = pd.DataFrame(results)

    print(results_df.to_string(index=False))

    passed = (results_df["status"] == "PASS").sum()
    failed = (results_df["status"] == "FAIL").sum()

    print(f"\nPassed: {passed}/{len(results_df)}")
    print(f"Failed: {failed}/{len(results_df)}")

    return results_df
# --------------------------------------------------
# LOGIN TIME DETECTOR
# --------------------------------------------------

def calculate_circular_time_difference(time1, time2):

    difference = abs(time1 - time2)

    return min(difference, 1440 - difference)


def detect_unseen_computer(
    daily_features,
    user,
    target_date,
    window_days=30,
    minimum_observations=5
):
    """
    Detect computers used by an employee on the target date
    that were not observed in their historical activity.

    A new computer is treated as a potential anomaly,
    not proof of malicious activity.
    """

    target_date = pd.Timestamp(target_date).normalize()

    window_start = target_date - pd.Timedelta(days=window_days)

    employee_data = daily_features[
        daily_features["user"] == user
    ].copy()

    # Separate historical activity from target-day activity
    history = employee_data[
        (employee_data["calendar_date"] >= window_start)
        & (employee_data["calendar_date"] < target_date)
    ].copy()

    target = employee_data[
        employee_data["calendar_date"] == target_date
    ].copy()

    # Check whether sufficient historical observations exist
    active_history = history[
        history["logon_events"] > 0
    ]

    observation_count = len(active_history)

    if observation_count < minimum_observations:
        return {
            "status": "BASELINE_UNAVAILABLE",
            "observations": observation_count,
            "known_computers": [],
            "observed_computers": [],
            "unseen_computers": [],
            "explanation": "Insufficient historical activity to establish a device baseline."
        }

    if target.empty:
        return {
            "status": "UNKNOWN",
            "observations": observation_count,
            "known_computers": [],
            "observed_computers": [],
            "unseen_computers": [],
            "explanation": "No target-day activity was available."
        }

    # Collect historical computer identifiers
    known_computers = set()

    for computers in history["computers_used"]:
        if isinstance(computers, (list, tuple, set)):
            known_computers.update(
                str(pc) for pc in computers
                if pd.notna(pc)
            )

    # Collect target-day computer identifiers
    observed_computers = set()

    for computers in target["computers_used"]:
        if isinstance(computers, (list, tuple, set)):
            observed_computers.update(
                str(pc) for pc in computers
                if pd.notna(pc)
            )

    unseen_computers = observed_computers - known_computers

    status = (
        "POTENTIAL_ANOMALY"
        if unseen_computers
        else "NORMAL"
    )

    return {
        "status": status,
        "observations": observation_count,
        "known_computers": sorted(known_computers),
        "observed_computers": sorted(observed_computers),
        "unseen_computers": sorted(unseen_computers),
        "explanation": (
            "Target-day computer(s) were not observed in the historical baseline."
            if unseen_computers
            else "All target-day computers were previously observed."
        )
    }


def evaluate_unseen_computer(daily_features, manifest):

    print("\nEvaluating unseen-computer detection...")

    expected_results = {
        "SYN01": "NORMAL",
        "SYN02": "NORMAL",
        "SYN03": "NORMAL",
        "SYN04": "NORMAL",
        "SYN05": "NORMAL",
        "SYN06": "BASELINE_UNAVAILABLE",
        "SYN07": "BASELINE_UNAVAILABLE",
        "SYN08": "POTENTIAL_ANOMALY",
        "SYN09": "NORMAL",
        "SYN10": "UNKNOWN"
    }

    results = []

    for _, test in manifest.iterrows():

        test_id = test["test_id"]
        user = test["user"]

        telemetry_available = str(
            test["telemetry_available"]
        ).strip().lower() in {"true", "1", "yes"}

        if not telemetry_available or pd.isna(test["target_date"]):

            detection = {
                "status": "UNKNOWN",
                "known_computers": [],
                "observed_computers": [],
                "unseen_computers": [],
                "explanation": "Telemetry unavailable."
            }

        else:

            detection = detect_unseen_computer(
                daily_features=daily_features,
                user=user,
                target_date=test["target_date"]
            )

        expected = expected_results[test_id]
        actual = detection["status"]

        results.append({
            "test_id": test_id,
            "user": user,
            "expected": expected,
            "actual": actual,
            "known_computers": detection["known_computers"],
            "observed_computers": detection["observed_computers"],
            "unseen_computers": detection["unseen_computers"],
            "status": actual,
            "result": "PASS" if expected == actual else "FAIL"
        })

    results_df = pd.DataFrame(results)

    print(
        results_df[
            [
                "test_id",
                "expected",
                "actual",
                "known_computers",
                "observed_computers",
                "unseen_computers",
                "result"
            ]
        ].to_string(index=False)
    )

    passed = (results_df["result"] == "PASS").sum()
    failed = (results_df["result"] == "FAIL").sum()

    print(f"\nUnseen-computer tests passed: {passed}/10")
    print(f"Unseen-computer tests failed: {failed}/10")

    return results_df


def detect_login_time(
    daily_features,
    user,
    target_date,
    threshold=3.0,
    window_days=30,
    minimum_observations=5
):

    baseline = prepare_employee_baseline(
        daily_features=daily_features,
        user=user,
        target_date=target_date,
        feature="first_login_minutes",
        window_days=window_days,
        minimum_observations=minimum_observations
    )

    if not baseline["baseline_available"]:
        return {
            "status": "BASELINE_UNAVAILABLE",
            "deviation_minutes": None,
            "observed_minutes": None,
            "baseline_mean_minutes": None,
            "explanation": "Insufficient historical login-time observations."
        }

    history = daily_features[
        (daily_features["user"] == user)
        & (daily_features["calendar_date"] >= baseline["window_start"])
        & (
            daily_features["calendar_date"]
            < pd.Timestamp(target_date).normalize()
        )
        & (daily_features["logon_events"] > 0)
        & (daily_features["first_login_minutes"].notna())
    ].copy()

    if len(history) < minimum_observations:
        return {
            "status": "BASELINE_UNAVAILABLE",
            "deviation_minutes": None,
            "observed_minutes": None,
            "baseline_mean_minutes": None,
            "explanation": "Insufficient historical observations."
        }

    # Calculate the circular mean of historical login times.
    angles = history["first_login_minutes"] * (2 * 3.141592653589793 / 1440)

    sin_mean = (
        __import__("numpy").sin(angles).mean()
    )

    cos_mean = (
        __import__("numpy").cos(angles).mean()
    )

    import numpy as np

    mean_angle = np.arctan2(sin_mean, cos_mean)

    if mean_angle < 0:
        mean_angle += 2 * np.pi

    baseline_mean = (
        mean_angle * 1440 / (2 * np.pi)
    ) % 1440

    # Calculate circular distance from historical mean.
    historical_deviations = history["first_login_minutes"].apply(
        lambda value: calculate_circular_time_difference(
            value,
            baseline_mean
        )
    )

    # Use historical median absolute deviation.
    median_deviation = historical_deviations.median()

    # Identify target-day activity.
    target = daily_features[
        (daily_features["user"] == user)
        & (
            daily_features["calendar_date"]
            == pd.Timestamp(target_date).normalize()
        )
    ]

    if target.empty:
        return {
            "status": "UNKNOWN",
            "deviation_minutes": None,
            "observed_minutes": None,
            "baseline_mean_minutes": baseline_mean,
            "explanation": "No target-day login activity."
        }

    observed = target.iloc[0]["first_login_minutes"]

    deviation = calculate_circular_time_difference(
        observed,
        baseline_mean
    )

    # Robust anomaly threshold.
    minimum_deviation_minutes = 120

    if median_deviation == 0:

        status = (
            "POTENTIAL_ANOMALY"
            if deviation >= minimum_deviation_minutes
            else "NORMAL"
        )

    else:

        robust_score = deviation / median_deviation

        status = (
            "POTENTIAL_ANOMALY"
            if (
                deviation >= minimum_deviation_minutes
                and robust_score >= threshold
            )
            else "NORMAL"
        )

    return {
        "status": status,
        "deviation_minutes": deviation,
        "observed_minutes": observed,
        "baseline_mean_minutes": baseline_mean,
        "historical_median_deviation": median_deviation,
        "explanation": (
            f"Observed login differs from historical circular mean "
            f"by {deviation:.1f} minutes."
        )
    }
# --------------------------------------------------
# LOGIN FREQUENCY DETECTOR
# --------------------------------------------------

def detect_login_frequency(
    daily_features,
    user,
    target_date,
    threshold=3.0
):

    baseline = prepare_employee_baseline(
        daily_features=daily_features,
        user=user,
        target_date=target_date,
        feature="logon_events",
        window_days=30,
        minimum_observations=5
    )

    if not baseline["baseline_available"]:
        return {
            "status": "BASELINE_UNAVAILABLE",
            "z_score": None,
            "observed": None,
            "baseline_mean": baseline["mean"],
            "explanation": "Insufficient historical observations."
        }

    target = daily_features[
        (daily_features["user"] == user)
        & (
            daily_features["calendar_date"]
            == pd.Timestamp(target_date).normalize()
        )
    ]

    if target.empty:
        return {
            "status": "UNKNOWN",
            "z_score": None,
            "observed": None,
            "baseline_mean": baseline["mean"],
            "explanation": "No target-day activity record."
        }

    observed = target.iloc[0]["logon_events"]

    mean = baseline["mean"]
    std = baseline["std"]

    if std is None or pd.isna(std) or std == 0:

        if observed > mean:
            status = "POTENTIAL_ANOMALY"
        else:
            status = "NORMAL"

        z_score = None

    else:

        z_score = (observed - mean) / std

        status = (
            "POTENTIAL_ANOMALY"
            if z_score >= threshold
            else "NORMAL"
        )

    return {
        "status": status,
        "z_score": z_score,
        "observed": observed,
        "baseline_mean": mean,
        "explanation": (
            f"Observed {observed} logins; "
            f"historical active-day mean is {mean:.2f}."
        )
    }


def evaluate_login_frequency(daily_features, manifest):

    frequency_expected = {
        "SYN01": "NORMAL",
        "SYN02": "NORMAL",
        "SYN03": "POTENTIAL_ANOMALY",
        "SYN04": "NORMAL",
        "SYN05": "NORMAL",
        "SYN06": "BASELINE_UNAVAILABLE",
        "SYN07": "BASELINE_UNAVAILABLE",
        "SYN08": "NORMAL",
        "SYN09": "POTENTIAL_ANOMALY",
        "SYN10": "UNKNOWN"
    }

    results = []

    for _, test in manifest.iterrows():

        test_id = test["test_id"]
        user = test["user"]
        target_date = test["target_date"]

        expected = frequency_expected[test_id]

        telemetry_available = str(
            test["telemetry_available"]
        ).strip().lower() in {"true", "1", "yes"}

        if (
            not telemetry_available
            or pd.isna(target_date)
        ):
            detection = {
                "status": "UNKNOWN",
                "observed": None,
                "baseline_mean": None,
                "z_score": None
            }

        else:
            detection = detect_login_frequency(
                daily_features=daily_features,
                user=user,
                target_date=target_date
            )

        actual = detection["status"]

        results.append({
            "test_id": test_id,
            "user": user,
            "expected": expected,
            "actual": actual,
            "observed_logins": detection.get("observed"),
            "baseline_mean": detection.get("baseline_mean"),
            "z_score": detection.get("z_score"),
            "status": "PASS" if expected == actual else "FAIL"
        })

    results_df = pd.DataFrame(results)

    print("\n========== LOGIN FREQUENCY EVALUATION ==========")

    print(
        results_df.to_string(
            index=False,
            na_rep="N/A"
        )
    )

    passed = (results_df["status"] == "PASS").sum()
    failed = (results_df["status"] == "FAIL").sum()

    print(f"\nPassed: {passed}/{len(results_df)}")
    print(f"Failed: {failed}/{len(results_df)}")

    return results_df



def evaluate_login_time(daily_features, manifest):
    expected_results = {
        "SYN01": "NORMAL",
        "SYN02": "POTENTIAL_ANOMALY",
        "SYN03": "NORMAL",
        "SYN04": "NORMAL",
        "SYN05": "POTENTIAL_ANOMALY",
        "SYN06": "BASELINE_UNAVAILABLE",
        "SYN07": "BASELINE_UNAVAILABLE",
        "SYN08": "NORMAL",
        "SYN09": "POTENTIAL_ANOMALY",
        "SYN10": "UNKNOWN"
    }

    results = []

    for _, test in manifest.iterrows():
        test_id = test["test_id"]
        user = test["user"]
        target_date = test["target_date"]

        telemetry_available = str(
            test["telemetry_available"]
        ).strip().lower() in {"true", "1", "yes"}

        if not telemetry_available or pd.isna(target_date):
            detection = {
                "status": "UNKNOWN",
                "deviation_minutes": None,
                "observed_minutes": None,
                "baseline_mean_minutes": None,
                "explanation": "Login telemetry is unavailable."
            }
        else:
            detection = detect_login_time(
                daily_features=daily_features,
                user=user,
                target_date=target_date
            )

        expected = expected_results[test_id]
        actual = detection["status"]

        results.append({
            "test_id": test_id,
            "user": user,
            "expected": expected,
            "actual": actual,
            "observed_minutes": detection.get("observed_minutes"),
            "baseline_mean_minutes": detection.get("baseline_mean_minutes"),
            "deviation_minutes": detection.get("deviation_minutes"),
            "status": "PASS" if actual == expected else "FAIL"
        })

    results_df = pd.DataFrame(results)

    print("\nLOGIN TIME DETECTOR EVALUATION")
    print("=" * 100)
    print(results_df.to_string(index=False))

    passed = (results_df["status"] == "PASS").sum()
    total = len(results_df)

    print(f"\nLogin-time evaluation: {passed}/{total} tests passed.")

    return results_df

def build_unified_detection_output(
    frequency_results,
    login_time_results,
    device_results,
    manifest
):
    print("\n========== UNIFIED DETECTION OUTPUT ==========")

    # Keep only the required frequency evidence
    frequency = frequency_results[
        [
            "test_id",
            "user",
            "actual",
            "observed_logins",
            "baseline_mean",
            "z_score",
            "status"
        ]
    ].copy()

    frequency = frequency.rename(columns={
        "actual": "login_frequency_status",
        "status": "frequency_test_result"
    })

    # Keep only the required login-time evidence
    login_time = login_time_results[
        [
            "test_id",
            "user",
            "actual",
            "observed_minutes",
            "baseline_mean_minutes",
            "deviation_minutes",
            "status"
        ]
    ].copy()

    login_time = login_time.rename(columns={
        "actual": "login_time_status",
        "status": "login_time_test_result"
    })

    # Keep only the required device evidence
    device = device_results[
        [
            "test_id",
            "user",
            "actual",
            "known_computers",
            "observed_computers",
            "unseen_computers",
            "result"
        ]
    ].copy()

    device = device.rename(columns={
        "actual": "device_status",
        "result": "device_test_result"
    })

    # Merge all three detector outputs
    unified = frequency.merge(
        login_time,
        on=["test_id", "user"],
        how="inner",
        validate="one_to_one"
    )

    unified = unified.merge(
        device,
        on=["test_id", "user"],
        how="inner",
        validate="one_to_one"
    )

    # Attach target dates from the original manifest
    unified = unified.merge(
        manifest[["test_id", "target_date"]],
        on="test_id",
        how="left",
        validate="one_to_one"
    )

    # Put identifying information first
    column_order = [
        "test_id",
        "user",
        "target_date",
        "login_frequency_status",
        "login_time_status",
        "device_status",
        "observed_logins",
        "baseline_mean",
        "z_score",
        "observed_minutes",
        "baseline_mean_minutes",
        "deviation_minutes",
        "known_computers",
        "observed_computers",
        "unseen_computers",
        "frequency_test_result",
        "login_time_test_result",
        "device_test_result"
    ]

    unified = unified[column_order]

    print(unified.to_string(index=False))

    print("\nUNIFIED OUTPUT COLUMN VALIDATION:")
    print(unified.columns.tolist())

    print(f"\nUnified records: {len(unified)}")

    # Validate record count and unique test IDs
    assert len(unified) == len(manifest), ("Unified output record count does not match manifest.")

    assert unified["test_id"].is_unique, ("Duplicate test IDs found in unified output.")

    assert unified["user"].notna().all(), ("Missing employee IDs in unified output.")   

    print("\nUNIFIED OUTPUT VALIDATION: PASSED")

    return unified


def build_employee_day_risk_output(unified_results):

    print("\n========== EMPLOYEE-DAY RISK AGGREGATION ==========")

    risk_results = unified_results.copy()

    assessable_statuses = {"NORMAL", "POTENTIAL_ANOMALY"}

    risk_levels = []
    coverage_levels = []
    assessed_counts = []
    behavioural_statuses = []
    device_statuses = []
    anomaly_evidence = []
    risk_explanations = []
    assessment_confidences = []

    for _, row in risk_results.iterrows():

        frequency_status = row["login_frequency_status"]
        time_status = row["login_time_status"]
        device_status = row["device_status"]

        # -----------------------------------------
        # 1. Behavioural evidence aggregation
        # -----------------------------------------

        behavioural_signals = [
            frequency_status,
            time_status
        ]

        if "POTENTIAL_ANOMALY" in behavioural_signals:
            behavioural_result = "ANOMALY"

        elif any(
            status in assessable_statuses
            for status in behavioural_signals
        ):
            behavioural_result = "NORMAL"

        elif "BASELINE_UNAVAILABLE" in behavioural_signals:
            behavioural_result = "BASELINE_UNAVAILABLE"

        else:
            behavioural_result = "UNKNOWN"

        # -----------------------------------------
        # 2. Device evidence
        # -----------------------------------------

        if device_status == "POTENTIAL_ANOMALY":
            device_result = "ANOMALY"

        elif device_status == "NORMAL":
            device_result = "NORMAL"

        else:
            device_result = device_status

        # -----------------------------------------
        # 3. Count assessable detectors
        # -----------------------------------------

        detector_statuses = [
            frequency_status,
            time_status,
            device_status
        ]

        assessed_count = sum(
            status in assessable_statuses
            for status in detector_statuses
        )

        if assessed_count == 3:
            coverage = "FULL"

        elif assessed_count > 0:
            coverage = "PARTIAL"

        else:
            coverage = "NONE"

        # -----------------------------------------
        # 4. Identify anomaly evidence
        # -----------------------------------------

        evidence = []

        if frequency_status == "POTENTIAL_ANOMALY":
            evidence.append("HIGH_LOGIN_FREQUENCY")

        if time_status == "POTENTIAL_ANOMALY":
            evidence.append("UNUSUAL_LOGIN_TIME")

        if device_status == "POTENTIAL_ANOMALY":
            evidence.append("UNSEEN_COMPUTER")

        # -----------------------------------------
        # 5. Assign provisional risk level
        # -----------------------------------------

        if (
            behavioural_result == "ANOMALY"
            and device_result == "ANOMALY"
        ):
            risk_level = "HIGH"

        elif (
            behavioural_result == "ANOMALY"
            or device_result == "ANOMALY"
        ):
            risk_level = "MODERATE"

        elif assessed_count > 0:
            risk_level = "LOW"

        else:
            risk_level = "INSUFFICIENT_DATA"
        # -----------------------------------------
        # 6. Assessment confidence
        # -----------------------------------------

        if coverage == "FULL":
            assessment_confidence = "HIGH"

        elif coverage == "PARTIAL":
            assessment_confidence = "LIMITED"

        else:
            assessment_confidence = "UNAVAILABLE"

        # -----------------------------------------
        # 7. Generate risk explanation
        # -----------------------------------------

        if risk_level == "HIGH":
            risk_explanation = (
                "Anomalies detected in both behavioural "
                "and device evidence domains. "
                "Further investigation is recommended."
            )

        elif risk_level == "MODERATE":

            explanation_parts = []

            if behavioural_result == "ANOMALY":
                explanation_parts.append(
                    "Behavioural deviation detected"
                )

            if device_result == "ANOMALY":
                explanation_parts.append(
                    "Unseen computer detected"
                )

            risk_explanation = (
                "; ".join(explanation_parts)
                + ". Further investigation may be warranted."
            )

        elif risk_level == "LOW":
            risk_explanation = (
                "No anomalies detected by the assessable "
                "detectors for this employee-day."
            )

        else:
            risk_explanation = (
                "Insufficient assessable telemetry or "
                "baseline information to classify risk."
            )
        risk_levels.append(risk_level)
        coverage_levels.append(coverage)
        assessed_counts.append(assessed_count)
        behavioural_statuses.append(behavioural_result)
        device_statuses.append(device_result)
        anomaly_evidence.append(evidence)
        risk_explanations.append(risk_explanation)
        assessment_confidences.append(assessment_confidence)
    # -----------------------------------------
    # 6. Add aggregation results
    # -----------------------------------------

    risk_results["behavioural_status"] = behavioural_statuses
    risk_results["aggregated_device_status"] = device_statuses
    risk_results["risk_level"] = risk_levels
    risk_results["data_coverage"] = coverage_levels
    risk_results["assessed_detectors"] = assessed_counts
    risk_results["anomaly_evidence"] = anomaly_evidence
    risk_results["risk_explanation"] = risk_explanations
    risk_results["assessment_confidence"] = assessment_confidences
    # -----------------------------------------
    # 7. Validate output
    # -----------------------------------------

    assert len(risk_results) == len(unified_results), (
        "Risk aggregation changed the number of records."
    )

    assert risk_results["test_id"].is_unique, (
        "Duplicate test IDs found in risk output."
    )

    assert risk_results["risk_level"].notna().all(), (
        "Missing risk classifications found."
    )

    print(
        risk_results[
        [
            "test_id",
            "user",
            "target_date",
            "behavioural_status",
            "aggregated_device_status",
            "risk_level",
            "data_coverage",
            "assessment_confidence",
            "assessed_detectors",
            "anomaly_evidence",
            "risk_explanation"
        ]
        ].to_string(index=False)
    )

    print("\nRISK LEVEL DISTRIBUTION:")
    print(risk_results["risk_level"].value_counts().to_string())

    print("\nRISK AGGREGATION VALIDATION: PASSED")

    return risk_results

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    try:
        events, manifest = load_test_data()

        if not validate_test_data(events, manifest):
            sys.exit(1)

        daily_features = build_daily_features(events)

        print("\nSample daily features:")
        print(daily_features.head(10).to_string(index=False))

        print("\nDaily feature engineering completed.")

        features_valid = validate_synthetic_features(
            daily_features, manifest
        )

        if not features_valid:
            sys.exit(1)

        print("\n========== BASELINE FUNCTION TEST ==========")

        test_baseline = prepare_employee_baseline(
            daily_features=daily_features,
            user="SYN_EMP_001",
            target_date=pd.Timestamp("2026-01-31"),
            feature="logon_events"
        )

        for key, value in test_baseline.items():
            print(f"{key}: {value}")

        frequency_results = evaluate_login_frequency(
            daily_features,
            manifest
        )

        if (frequency_results["status"] == "FAIL").any():
            sys.exit(1)

        baseline_results = test_baseline_availability(
            daily_features, manifest
        )

        if (baseline_results["status"] == "FAIL").any():
            sys.exit(1)
        print("\n========== LOGIN FREQUENCY DETECTOR TEST ==========")

        frequency_test = detect_login_frequency(
            daily_features,
            user="SYN_EMP_003",
            target_date="2026-01-31"
        )

        for key, value in frequency_test.items():
            print(f"{key}: {value}")
        print("\n========== LOGIN TIME DETECTOR TEST ==========")

        time_test = detect_login_time(
            daily_features=daily_features,
            user="SYN_EMP_002",
            target_date="2026-01-31"
        )

        for key, value in time_test.items():
            print(f"{key}: {value}")
        login_time_results = evaluate_login_time(
            daily_features=daily_features,
            manifest=manifest
        )

        if (login_time_results["status"] == "FAIL").any():
            print("\nLOGIN TIME EVALUATION FAILED.")
            sys.exit(1)

        device_results = evaluate_unseen_computer(
            daily_features=daily_features,
            manifest=manifest
        )

        if (device_results["result"] == "FAIL").any():
            print("\nUNSEEN-COMPUTER EVALUATION FAILED.")
            sys.exit(1)
        

        unified_results = build_unified_detection_output(
            frequency_results=frequency_results,
            login_time_results=login_time_results,
            device_results=device_results,
            manifest=manifest
        )
        risk_results = build_employee_day_risk_output(
            unified_results
        )
    except Exception as error:
        print(f"\nRUNNER ERROR: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()