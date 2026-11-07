import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

FOLDERS = [
    PROJECT_DIR / "answers"
]
for folder in FOLDERS:

    print("\n" + "=" * 80)
    print("FOLDER:", folder)
    print("=" * 80)

    if not folder.exists():
        print("Folder does not exist.")
        continue

    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        print("No CSV files found.")
        continue

    for file in csv_files:

        print("\n" + "-" * 70)
        print("FILE:", file.name)

        try:
            df = pd.read_csv(file)

            print("Shape:", df.shape)
            print("Columns:", df.columns.tolist())

            print("\nFirst 5 records:")
            print(df.head().to_string(index=False))

            print("\nData types:")
            print(df.dtypes)

        except Exception as error:
            print("Could not read file:", error)