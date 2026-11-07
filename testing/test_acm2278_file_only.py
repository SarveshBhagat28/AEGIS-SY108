import pandas as pd

df = pd.read_csv("data/unified_activity_features.csv")
df["calendar_date"] = pd.to_datetime(df["calendar_date"])

data = df[
    (df["user"] == "ACM2278")
    & (df["calendar_date"] >= "2010-07-19")
    & (df["calendar_date"] <= "2010-08-24")
    & (df["file_day_covered"] == True)
].copy()

data["removable_total"] = (
    data["to_removable_events"] + data["from_removable_events"]
)

data["file_modification_total"] = (
    data["file_copies"] + data["file_writes"]
)

historical = data[data["calendar_date"] < "2010-08-18"]
incident = data[data["calendar_date"] >= "2010-08-18"]

metrics = [
    "removable_total",
    "file_modification_total",
    "file_deletes"
]

print("COVERED DAYS")
print("Historical:", len(historical))
print("Incident:", len(incident))

print("\nHISTORICAL SUMMARY")
print(historical[metrics].agg(["mean", "median", "max"]))

print("\nINCIDENT SUMMARY")
print(incident[metrics].agg(["mean", "median", "max"]))

print("\nHISTORICAL DAILY ACTIVITY")
print(historical[["calendar_date"] + metrics].to_string(index=False))

print("\nINCIDENT DAILY ACTIVITY")
print(incident[["calendar_date"] + metrics].to_string(index=False))