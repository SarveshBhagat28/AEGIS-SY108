import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

insiders = pd.read_csv(BASE_DIR / "answers" / "insiders.csv")

insiders["dataset"] = pd.to_numeric(
    insiders["dataset"], errors="coerce"
)

insiders = insiders[insiders["dataset"] == 6.2].copy()

datasets = {
    "logon": "logon.csv",
    "file": "file.csv",
    "device": "device.csv"
}

for dataset_name, filename in datasets.items():

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} COVERAGE")
    print("=" * 70)

    df = pd.read_csv(
        BASE_DIR / filename,
        usecols=["date", "user"]
    )

    df["date"] = pd.to_datetime(
        df["date"],
        format="%m/%d/%Y %H:%M:%S",
        errors="coerce"
    )

    for _, insider in insiders.iterrows():

        user = insider["user"]

        employee_data = df[df["user"] == user]

        print(f"\nEmployee: {user}")
        print("Total events:", len(employee_data))

        if not employee_data.empty:
            print("First event:", employee_data["date"].min())
            print("Last event:", employee_data["date"].max())

            print(
                "Events during incident:",
                (
                    (employee_data["date"] >= pd.to_datetime(
                        insider["start"], errors="coerce"
                    ))
                    &
                    (employee_data["date"] <= pd.to_datetime(
                        insider["end"], errors="coerce"
                    ))
                ).sum()
            )