
import pandas as pd
import numpy as np


DEFAULT_WINDOW_DAYS = 30
DEFAULT_MIN_OBSERVATIONS = 5


def build_removable_baseline(
    history,
    min_observations=DEFAULT_MIN_OBSERVATIONS
):
    required = {
        "file_day_covered",
        "to_removable_events",
        "from_removable_events"
    }

    missing = required - set(history.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    covered = history[
        history["file_day_covered"].fillna(False).astype(bool)
    ]

    result = {}

    for metric in [
        "to_removable_events",
        "from_removable_events"
    ]:
        values = pd.to_numeric(
            covered[metric],
            errors="coerce"
        ).dropna()

        if len(values) < min_observations:
            result[metric] = {
                "available": False,
                "observations": len(values)
            }
            continue

        median = float(values.median())
        mad = float((values - median).abs().median())
        p95 = float(values.quantile(0.95))

        result[metric] = {
            "available": True,
            "observations": len(values),
            "median": median,
            "mad": mad,
            "p95": p95,
            "threshold": max(p95, median + 3 * mad)
        }

    return result


def evaluate_removable_activity(
    daily_data,
    employee,
    target_date,
    window_days=DEFAULT_WINDOW_DAYS,
    min_observations=DEFAULT_MIN_OBSERVATIONS
):
    target_date = pd.Timestamp(target_date).normalize()

    data = daily_data.copy()
    data["calendar_date"] = pd.to_datetime(
        data["calendar_date"]
    ).dt.normalize()

    employee_data = data[data["user"] == employee]

    current = employee_data[
        employee_data["calendar_date"] == target_date
    ]

    if current.empty:
        return {
            "status": "INSUFFICIENT_DATA",
            "severity": "UNKNOWN",
            "reason": "No employee-day record exists."
        }

    current = current.iloc[0]

    if pd.isna(current["file_day_covered"]) or not bool(
        current["file_day_covered"]
    ):
        return {
            "status": "INSUFFICIENT_DATA",
            "severity": "UNKNOWN",
            "reason": "File telemetry is not covered."
        }

    window_start = target_date - pd.Timedelta(
        days=window_days
    )

    history = employee_data[
        (employee_data["calendar_date"] >= window_start)
        & (employee_data["calendar_date"] < target_date)
    ]

    baseline = build_removable_baseline(
        history,
        min_observations
    )

    findings = []
    unavailable = []

    for metric in [
        "to_removable_events",
        "from_removable_events"
    ]:
        profile = baseline[metric]
        value = pd.to_numeric(
            pd.Series([current[metric]]),
            errors="coerce"
        ).iloc[0]

        if not profile["available"] or pd.isna(value):
            unavailable.append(metric)
            continue

        if value > profile["threshold"]:
            findings.append({
                "metric": metric,
                "observed": int(value),
                "threshold": round(
                    profile["threshold"], 2
                ),
                "baseline_median": profile["median"],
                "baseline_p95": profile["p95"]
            })

    if unavailable:
        return {
            "status": "INSUFFICIENT_DATA",
            "severity": "UNKNOWN",
            "reason": "Insufficient baseline or current data.",
            "unavailable_metrics": unavailable,
            "findings": findings
        }

    if findings:
        return {
            "status": "ANOMALOUS",
            "severity": "MODERATE",
            "findings": findings,
            "baseline": baseline
        }

    return {
        "status": "NORMAL",
        "severity": "LOW",
        "findings": [],
        "baseline": baseline
    }

