
import pandas as pd

df = pd.read_csv("data/unified_activity_features.csv")
df["calendar_date"] = pd.to_datetime(df["calendar_date"])

data = df[
    (df["user"] == "CDE1846")
    & (df["calendar_date"] >= "2011-01-22")
    & (df["calendar_date"] <= "2011-04-25")
].copy()

historical = data[data["calendar_date"] < "2011-02-21"]
incident = data[
    (data["calendar_date"] >= "2011-02-21")
    & (data["calendar_date"] <= "2011-04-25")
]

for name, period in [("HISTORICAL", historical), ("INCIDENT", incident)]:
    print(f"\n{name}")
    print("Employee-day records:", len(period))
    print("File-covered days:", period["file_day_covered"].sum())
    print("Device-covered days:", period["device_day_covered"].sum())
    print("Both covered:", (
        period["file_day_covered"] & period["device_day_covered"]
    ).sum())
