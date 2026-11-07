import pandas as pd

df = pd.read_csv("data/evaluation_daily_labels.csv")
df["date"] = pd.to_datetime(df["date"])

data = df[df["user"] == "CDE1846"]

print(data[[
    "user", "date", "scenario", "known_insider",
    "incident_start", "incident_end",
    "file_day_covered", "device_day_covered"
]].to_string(index=False))