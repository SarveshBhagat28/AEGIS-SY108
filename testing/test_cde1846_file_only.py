
import pandas as pd

df = pd.read_csv("data/unified_activity_features.csv")
df["calendar_date"] = pd.to_datetime(df["calendar_date"])

data = df[
    (df["user"] == "CDE1846")
    & (df["calendar_date"] >= "2011-01-22")
    & (df["calendar_date"] <= "2011-04-25")
    & (df["file_day_covered"] == True)
].copy()

data["removable_total"] = (
    data["to_removable_events"] + data["from_removable_events"]
)

data["file_modification_total"] = (
    data["file_copies"] + data["file_writes"]
)

historical = data[data["calendar_date"] < "2011-02-21"]
incident = data[data["calendar_date"] >= "2011-02-21"]

metrics = [
    "removable_total",
    "file_modification_total",
    "file_deletes"
]

print("HISTORICAL DAYS:", len(historical))
print("INCIDENT DAYS:", len(incident))

print("\nHISTORICAL ACTIVITY")
print(historical[["calendar_date"] + metrics].to_string(index=False))

print("\nINCIDENT ACTIVITY")
print(incident[["calendar_date"] + metrics].to_string(index=False))
