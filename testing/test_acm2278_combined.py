
import pandas as pd

df = pd.read_csv("data/evaluation_daily_labels.csv")

data = df[df["user"] == "CDE1846"]

print("CDE1846 INCIDENT DETAILS")
print(data[[
    "user",
    "scenario",
    "known_insider",
    "incident_start",
    "incident_end"
]].drop_duplicates().to_string(index=False))
