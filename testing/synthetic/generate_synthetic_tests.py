import pandas as pd
import numpy as np
from pathlib import Path


# ==========================================================
# CONFIGURATION
# ==========================================================

OUTPUT_DIR = Path(
    r"D:\Programs\AEGIS 2\testing\synthetic"
)

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

START_DATE = pd.Timestamp("2026-01-01")

EVENT_COLUMNS = [
    "id",
    "date",
    "user",
    "pc",
    "activity"
]


# ==========================================================
# DATA GENERATION
# ==========================================================

events = []
test_cases = []

event_id = 1


def add_event(user, timestamp, pc="PC001",
              activity="Logon"):

    global event_id

    events.append({
        "id": event_id,
        "date": timestamp,
        "user": user,
        "pc": pc,
        "activity": activity
    })

    event_id += 1


def add_normal_history(user, days=30):



    for day in range(days):

        current_date = START_DATE + pd.Timedelta(days=day)

        # Normally 1 or 2 login events per day.
        login_count = int(rng.choice([1, 1, 1, 2]))

        for login_number in range(login_count):

            login_time = current_date + pd.Timedelta(
                hours=9,
                minutes=int(rng.integers(-15, 16))
            )

            if login_number == 1:
                login_time += pd.Timedelta(hours=2)

            add_event(user, login_time)

        logout_time = current_date + pd.Timedelta(
            hours=17,
            minutes=int(rng.integers(-20, 21))
        )

        add_event(
            user,
            logout_time,
            activity="Logoff"
        )


def add_test_case(
    test_id,
    user,
    description,
    expected,
    target_date=None,
    test_feature="unspecified",
    telemetry_available=True,
    logon_telemetry_available=True,
    device_telemetry_available=True
):
    test_cases.append({
        "test_id": test_id,
        "user": user,
        "description": description,
        "target_date": (
            target_date.strftime("%Y-%m-%d")
            if target_date is not None else None
        ),
        "test_feature": test_feature,
        "expected_result": expected,
        "telemetry_available": telemetry_available,
        "logon_telemetry_available": logon_telemetry_available,
        "device_telemetry_available": device_telemetry_available
    })

# ==========================================================
# SYN01 - NORMAL LOGIN
# ==========================================================

user = "SYN_EMP_001"

add_normal_history(user)

add_event(
    user,
    START_DATE + pd.Timedelta(days=30, hours=9)
)

add_test_case(
    "SYN01",
    user,
    "Consistent login around 9 AM",
    "NORMAL",
    target_date=START_DATE + pd.Timedelta(days=30),
    test_feature="normal_behavior"

)


# ==========================================================
# SYN02 - AFTER-HOURS LOGIN
# ==========================================================

user = "SYN_EMP_002"

add_normal_history(user)

add_event(
    user,
    START_DATE + pd.Timedelta(days=30, hours=2)
)

add_test_case(
    "SYN02",
    user,
    "Unusual login at 2 AM",
    "POTENTIAL_ANOMALY",
    target_date=START_DATE + pd.Timedelta(days=30),
    test_feature="login_time"
)

user = "SYN_EMP_003"

# SYN03: Sudden increase in login frequency
add_normal_history(user)

test_date = START_DATE + pd.Timedelta(days=30)

for hour in [8, 9, 10, 11, 12, 13, 14, 15]:
    add_event(
        user,
        test_date + pd.Timedelta(hours=hour)
    )

add_test_case(
    "SYN03",
    user,
    "Sudden increase in login frequency",
    "POTENTIAL_ANOMALY",
    target_date=START_DATE + pd.Timedelta(days=30),
    test_feature="login_frequency"
)


# ==========================================================
# SYN04 - CONSISTENT ACTIVITY
# ==========================================================

user = "SYN_EMP_004"

add_normal_history(user, days=60)

add_event(
    user,
    START_DATE + pd.Timedelta(days=60, hours=9)
)

add_test_case(
    "SYN04",
    user,
    "Consistent login behaviour over 60 days",
    "NORMAL",
    target_date=START_DATE + pd.Timedelta(days=60),
    test_feature="normal_behavior"
)


# ==========================================================
# SYN05 - REPEATED UNUSUAL LOGIN TIMES
# ==========================================================

user = "SYN_EMP_005"

add_normal_history(user)

for day in range(30, 35):

    add_event(
        user,
        START_DATE + pd.Timedelta(days=day, hours=2)
    )

add_test_case(
    "SYN05",
    user,
    "Repeated after-hours logins",
    "POTENTIAL_ANOMALY",
    target_date=START_DATE + pd.Timedelta(days=34),
    test_feature="repeated_login_time"
)


# ==========================================================
# SYN06 - INSUFFICIENT HISTORICAL OBSERVATIONS
# ==========================================================

user = "SYN_EMP_006"

for day in range(3):

    add_event(
        user,
        START_DATE + pd.Timedelta(days=day, hours=9)
    )

add_event(
    user,
    START_DATE + pd.Timedelta(days=3, hours=2)
)

add_test_case(
    "SYN06",
    user,
    "Three historical active days followed by a target login",
    "BASELINE_UNAVAILABLE",
    target_date=START_DATE + pd.Timedelta(days=3),
    test_feature="baseline_availability"
)


# ==========================================================
# SYN07 - NEW EMPLOYEE
# ==========================================================

user = "SYN_EMP_007"

add_event(
    user,
    START_DATE + pd.Timedelta(days=30, hours=2)
)

add_test_case(
    "SYN07",
    user,
    "Employee without historical activity",
    "BASELINE_UNAVAILABLE",
    target_date=START_DATE + pd.Timedelta(days=30),
    test_feature="baseline_availability"
)


# ==========================================================
# SYN08 - LOGIN FROM ANOTHER COMPUTER
# ==========================================================

user = "SYN_EMP_008"

add_normal_history(user)

add_event(
    user,
    START_DATE + pd.Timedelta(days=30, hours=9),
    pc="PC999"
)

add_test_case(
    "SYN08",
    user,
    "Login from a previously unseen computer",
    "POTENTIAL_ANOMALY",
    target_date=START_DATE + pd.Timedelta(days=30),
    test_feature="unseen_computer"
)

# ==========================================================
# SYN09 - ONE-DAY UNUSUAL ACTIVITY
# ==========================================================

user = "SYN_EMP_009"

add_normal_history(user)

test_date = START_DATE + pd.Timedelta(days=30)

for hour in [1, 2, 3]:

    add_event(
        user,
        test_date + pd.Timedelta(hours=hour)
    )

add_test_case(
    "SYN09",
    user,
    "One-day burst of unusual login activity",
    "POTENTIAL_ANOMALY",
    target_date=START_DATE + pd.Timedelta(days=30),
    test_feature="combined_time_frequency"
)


# ==========================================================
# SYN10 - MISSING TELEMETRY
# ==========================================================

user = "SYN_EMP_010"

add_test_case(
    "SYN10",
    user,
    "Login telemetry unavailable",
    "UNKNOWN",
    target_date=None,
    test_feature="telemetry_availability",
    telemetry_available=False,
    logon_telemetry_available=False,
    device_telemetry_available=False
)

# ==========================================================
# SAVE DATASETS
# ==========================================================

event_df = pd.DataFrame(
    events,
    columns=EVENT_COLUMNS
)

test_df = pd.DataFrame(test_cases)

event_df = event_df.sort_values(
    ["user", "date"]
).reset_index(drop=True)

event_df.to_csv(
    OUTPUT_DIR / "synthetic_logon_events.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "synthetic_test_cases.csv",
    index=False
)


# ==========================================================
# SUMMARY
# ==========================================================

print("=" * 60)
print("AEGIS SYNTHETIC DATA GENERATION")
print("=" * 60)

print(f"Total events: {len(event_df):,}")
print(f"Unique employees: {event_df['user'].nunique()}")
print(f"Test scenarios: {len(test_df)}")

print("\nExpected test outcomes:")

print(
    test_df[
        ["test_id", "description", "expected_result"]
    ].to_string(index=False)
)

print("\nGenerated files:")

print(
    OUTPUT_DIR / "synthetic_logon_events.csv"
)

print(
    OUTPUT_DIR / "synthetic_test_cases.csv"
)

print("\nSYNTHETIC DATA GENERATION COMPLETED")