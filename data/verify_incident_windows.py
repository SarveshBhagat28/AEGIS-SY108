import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

insiders = pd.read_csv(BASE_DIR / "answers" / "insiders.csv")

insiders["dataset"] = pd.to_numeric(
    insiders["dataset"], errors="coerce"
)

insiders = insiders[insiders["dataset"] == 6.2].copy()

# Correct the malformed date for analysis only.
insiders.loc[
    insiders["user"] == "CDE1846", "start"
] = "02/21/2011 11:43:39"

datasets = ["logon", "file", "device"]

for dataset in datasets:

    df = pd.read_csv(
        BASE_DIR / f"{dataset}.csv",
        usecols=["date", "user"]
    )

    df["date"] = pd.to_datetime(
        df["date"],
        format="%m/%d/%Y %H:%M:%S",
        errors="coerce"
    )

    print("\n" + "=" * 75)
    print(f"{dataset.upper()} INCIDENT VERIFICATION")
    print("=" * 75)

    for _, insider in insiders.iterrows():

        user = insider["user"]

        start = pd.to_datetime(
            insider["start"],
            format="%m/%d/%Y %H:%M:%S",
            errors="coerce"
        )

        end = pd.to_datetime(
            insider["end"],
            format="%m/%d/%Y %H:%M:%S",
            errors="coerce"
        )

        employee_data = df[df["user"] == user].copy()

        incident_data = employee_data[
            (employee_data["date"] >= start) &
            (employee_data["date"] <= end)
        ]

        print(f"\nEmployee: {user}")
        print(f"Incident: {start} to {end}")
        print("Events during incident:", len(incident_data))

        if not incident_data.empty:
            print("\nFirst incident-period events:")
            print(
                incident_data.sort_values("date").head(5).to_string(
                    index=False
                )
            )

        print("\nEvents on incident end date:")
        if pd.notna(end):
            same_day = employee_data[
                employee_data["date"].dt.date == end.date()
            ]
            print(same_day.sort_values("date").to_string(index=False))