
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "file.csv"

OUTPUT_DIR = BASE_DIR / "testing" / "reports"

OUTPUT_FILE = (
    OUTPUT_DIR / "scenario1_acm2278_file_events.csv"
)

EMPLOYEE = "ACM2278"

INCIDENT_START = pd.Timestamp(
    "2010-08-18 21:47:42"
)

INCIDENT_END = pd.Timestamp(
    "2010-08-24 03:48:51"
)

LOOKBACK_DAYS = 30
FOLLOWUP_DAYS = 30
CHUNK_SIZE = 200_000


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pre_start = (
        INCIDENT_START - pd.Timedelta(days=LOOKBACK_DAYS)
    )

    post_end = (
        INCIDENT_END + pd.Timedelta(days=FOLLOWUP_DAYS)
    )

    extracted = []
    total_rows = 0
    matched_rows = 0

    print("SCENARIO 1 FILE EVENT EXTRACTION")
    print("=" * 65)
    print(f"Employee: {EMPLOYEE}")
    print(f"Pre-incident start: {pre_start}")
    print(f"Incident start: {INCIDENT_START}")
    print(f"Incident end: {INCIDENT_END}")
    print(f"Post-incident end: {post_end}")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):

        total_rows += len(chunk)

        # Normalize column names.
        chunk.columns = (
            chunk.columns.astype(str)
            .str.strip()
            .str.lower()
        )

        required = {"user", "date"}

        if not required.issubset(chunk.columns):

            raise ValueError(
                "Required columns missing. "
                f"Found columns: {chunk.columns.tolist()}"
            )

        chunk["user"] = (
            chunk["user"].astype(str).str.strip()
        )

        employee_rows = chunk[
            chunk["user"] == EMPLOYEE
        ].copy()

        if employee_rows.empty:
            print(
                f"Processed chunk {chunk_number:,} "
                f"| Total rows: {total_rows:,}"
            )
            continue

        employee_rows["event_timestamp"] = pd.to_datetime(
            employee_rows["date"],
            format="%m/%d/%Y %H:%M:%S",
            errors="coerce",
        )

        employee_rows = employee_rows[
            employee_rows["event_timestamp"].notna()
        ]

        employee_rows = employee_rows[
            (
                employee_rows["event_timestamp"] >= pre_start
            )
            &
            (
                employee_rows["event_timestamp"] <= post_end
            )
        ].copy()

        if employee_rows.empty:
            continue

        timestamp = employee_rows["event_timestamp"]

        employee_rows["period"] = "OUTSIDE_WINDOW"

        employee_rows.loc[
            timestamp < INCIDENT_START,
            "period",
        ] = "PRE_INCIDENT"

        employee_rows.loc[
            (
                timestamp >= INCIDENT_START
            )
            &
            (
                timestamp <= INCIDENT_END
            ),
            "period",
        ] = "INCIDENT"

        employee_rows.loc[
            timestamp > INCIDENT_END,
            "period",
        ] = "POST_INCIDENT"

        extracted.append(employee_rows)

        matched_rows += len(employee_rows)

        print(
            f"Processed chunk {chunk_number:,} "
            f"| Extracted events: {matched_rows:,}"
        )

    if not extracted:

        print("No matching records found.")
        return

    result = pd.concat(
        extracted,
        ignore_index=True,
    )

    result = result.sort_values(
        "event_timestamp"
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nEXTRACTION SUMMARY")
    print("=" * 65)

    print(f"Raw rows processed: {total_rows:,}")
    print(f"Matching events: {len(result):,}")

    print("\nEvents by period:")

    print(
        result["period"]
        .value_counts()
        .reindex(
            [
                "PRE_INCIDENT",
                "INCIDENT",
                "POST_INCIDENT",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print("\nEvents by activity:")

    if "activity" in result.columns:

        print(
            result["activity"]
            .value_counts(dropna=False)
            .to_string()
        )

    print("\nOutput:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
