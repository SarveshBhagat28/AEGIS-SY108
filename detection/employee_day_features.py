
import pandas as pd

from detection.logon_processor import load_cert_logon_data


def build_employee_day_features(events):
    """
    Convert raw CERT logon events into employee-day features.

    Each output row represents one employee on one calendar day.

    No anomaly labels or fixed working-hour assumptions
    are introduced during feature engineering.
    """

    required_columns = {
        "date",
        "user",
        "pc",
        "activity"
    }

    missing = required_columns - set(events.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = events.copy()

    # Ensure timestamps are valid.
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date", "user", "pc", "activity"]
    ).copy()

    # Extract calendar date.
    df["calendar_date"] = df["date"].dt.normalize()

    # Normalize activity labels.
    df["activity"] = (
        df["activity"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------
    # 1. General employee-day activity
    # --------------------------------------------------

    daily = (
        df.groupby(["user", "calendar_date"])
        .agg(
            event_count=("activity", "size"),
            unique_pcs=("pc", "nunique"),
            computers_used=(
                "pc",
                lambda values: "|".join(
                    sorted(values.dropna().unique())
                )
            )
        )
        .reset_index()
    )

    # --------------------------------------------------
    # 2. Login-specific features
    # --------------------------------------------------

    logons = df[
        df["activity"] == "logon"
    ].copy()

    login_daily = (
        logons.groupby(["user", "calendar_date"])
        .agg(
            logon_count=("date", "size"),
            first_login=("date", "min"),
            last_login=("date", "max")
        )
        .reset_index()
    )

    # --------------------------------------------------
    # 3. Logout-specific features
    # --------------------------------------------------

    logoffs = df[
        df["activity"] == "logoff"
    ]

    logoff_daily = (
        logoffs.groupby(["user", "calendar_date"])
        .size()
        .reset_index(name="logoff_count")
    )

    # --------------------------------------------------
    # 4. Merge employee-day features
    # --------------------------------------------------

    daily = daily.merge(
        login_daily,
        on=["user", "calendar_date"],
        how="left"
    )

    daily = daily.merge(
        logoff_daily,
        on=["user", "calendar_date"],
        how="left"
    )

    # Missing login/logout counts are zero.
    daily["logon_count"] = (
        daily["logon_count"]
        .fillna(0)
        .astype(int)
    )

    daily["logoff_count"] = (
        daily["logoff_count"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------
    # 5. Convert login timestamps to minutes after midnight
    # --------------------------------------------------

    daily["first_login_minutes"] = (
        daily["first_login"]
        .dt.hour * 60
        + daily["first_login"].dt.minute
    )

    daily["last_login_minutes"] = (
        daily["last_login"]
        .dt.hour * 60
        + daily["last_login"].dt.minute
    )

     # --------------------------------------------------
    # 6. Additional indicators
    # --------------------------------------------------

    daily["has_logon_activity"] = (
        daily["logon_count"] > 0
    )

    daily["has_logoff_activity"] = (
        daily["logoff_count"] > 0
    )

    # True means at least one event was recorded.
    daily["active_day"] = (
        daily["event_count"] > 0
    )

    daily = daily.sort_values(
        ["user", "calendar_date"]
    ).reset_index(drop=True)

    return daily

def main():

    file_path = r"D:\Programs\AEGIS 2\data\logon.csv"

    print("\nLoading CERT logon data...")

    events = load_cert_logon_data(file_path)

    print("\nBuilding employee-day features...")

    daily_features = build_employee_day_features(events)

    print("\n========== EMPLOYEE-DAY FEATURE REPORT ==========")

    print(
        f"Total employee-day records: {len(daily_features):,}"
    )

    print(
        f"Unique employees: {daily_features['user'].nunique():,}"
    )

    print(
        f"Date range: "
        f"{daily_features['calendar_date'].min()} "
        f"to "
        f"{daily_features['calendar_date'].max()}"
    )

    print("\nFeature columns:")

    print(daily_features.columns.tolist())

    print("\nSample records:")

    print(daily_features.head(10).to_string(index=False))

    print("\nLogin count statistics:")

    print(daily_features["logon_count"].describe())

    print("\nSaving employee-day features...")

    output_path = (
        r"D:\Programs\AEGIS 2\data\daily_activity_features.csv"
    )

    daily_features.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

