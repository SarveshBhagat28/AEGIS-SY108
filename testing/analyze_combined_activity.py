
import pandas as pd

df = pd.read_csv("data/unified_activity_features.csv")
df["calendar_date"] = pd.to_datetime(df["calendar_date"])

employee = df[
    (df["user"] == "CMP2946")
    & (df["calendar_date"] >= "2011-01-01")
    & (df["calendar_date"] < "2011-03-05")
].copy()

employee = employee[
    (employee["file_day_covered"] == True)
    & (employee["device_day_covered"] == True)
].copy()

employee["period"] = employee["calendar_date"].apply(
    lambda date: "Historical"
    if date < pd.Timestamp("2011-02-07")
    else "Incident"
)

metrics = [
    "to_removable_events",
    "from_removable_events",
    "file_copies",
    "file_writes",
    "file_deletes",
    "device_connects",
    "device_disconnects"
]

print("COVERED DAYS")
print(employee.groupby("period").size())

print("\nMETRIC SUMMARY")
print(
    employee.groupby("period")[metrics]
    .agg(["mean", "median", "max"])
    .round(2)
)

print("\nCO-OCCURRENCE COUNTS")

employee["removable_activity"] = (
    employee["to_removable_events"]
    + employee["from_removable_events"] > 0
)

employee["file_modification"] = (
    employee["file_copies"]
    + employee["file_writes"] > 0
)

employee["device_activity"] = (
    employee["device_connects"]
    + employee["device_disconnects"] > 0
)

combinations = {
    "Removable + file modification": (
        employee["removable_activity"]
        & employee["file_modification"]
    ),
    "Removable + device activity": (
        employee["removable_activity"]
        & employee["device_activity"]
    ),
    "File modification + device activity": (
        employee["file_modification"]
        & employee["device_activity"]
    ),
    "All three signal groups": (
        employee["removable_activity"]
        & employee["file_modification"]
        & employee["device_activity"]
    )
}

for name, condition in combinations.items():
    print("\n" + name)
    print(
        employee.assign(match=condition)
        .groupby("period")["match"]
        .sum()
    )

print("\nINCIDENT DAYS WITH MULTIPLE SIGNALS")

selected = employee[
    employee["removable_activity"]
    & employee["file_modification"]
    & employee["device_activity"]
]

print(
    selected[
        ["calendar_date"] + metrics
    ].to_string(index=False)
)
