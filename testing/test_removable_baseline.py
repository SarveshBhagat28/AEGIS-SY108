
import pandas as pd
from detection.removable_media_baseline import evaluate_removable_activity

df = pd.read_csv("data/unified_activity_features.csv")
df["calendar_date"] = pd.to_datetime(df["calendar_date"])

employee_days = df[
    (df["user"] == "CMP2946")
    & (df["calendar_date"] >= "2011-02-07")
    & (df["calendar_date"] < "2011-03-05")
    & (df["file_day_covered"] == True)
].copy()

for date in employee_days["calendar_date"].sort_values():
    current = employee_days[
        employee_days["calendar_date"] == date
    ].iloc[0]

    result = evaluate_removable_activity(
        df, "CMP2946", date
    )

    baseline = result.get("baseline", {})

    to_threshold = baseline.get(
        "to_removable_events", {}
    ).get("threshold")

    from_threshold = baseline.get(
        "from_removable_events", {}
    ).get("threshold")

    print(
        date.date(),
        "Outbound:", current["to_removable_events"],
        "Threshold:", to_threshold,
        "Inbound:", current["from_removable_events"],
        "Threshold:", from_threshold,
        "Status:", result["status"]
    )
