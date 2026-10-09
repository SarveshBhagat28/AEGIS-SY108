
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="A.E.G.I.S. | Insider Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ A.E.G.I.S.")
st.subheader("Anomaly Detection and Entity-based Guardian Intelligence System")
st.caption("CERT Insider Threat Analysis | Prototype")

st.info(
    "This prototype identifies unusual activity patterns. "
    "An anomaly is not proof of malicious behaviour."
)


logon_file = st.sidebar.file_uploader(
    "logon.csv — maximum 800 MB",
    type=["csv"],
    max_upload_size=800
)

device_file = st.sidebar.file_uploader(
    "device.csv — maximum 800 MB",
    type=["csv"],
    max_upload_size=800
)

users_file = st.sidebar.file_uploader(
    "users.csv — maximum 800 MB",
    type=["csv"],
    max_upload_size=800
)

file_file = st.sidebar.file_uploader(
    "file.csv — maximum 2 GB",
    type=["csv"],
    max_upload_size=2000
)

psych_file = st.sidebar.file_uploader(
    "psychometric.csv — maximum 800 MB",
    type=["csv"],
    max_upload_size=800
)

st.sidebar.caption(
    "HTTP and email data are excluded. Large file.csv can be added later."
)


def find_column(df, candidates):
    columns = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name in columns:
            return columns[name]
    return None


def load_activity(upload, source_name):
    if upload is None:
        return None

    df = pd.read_csv(upload, low_memory=False)
    st.write(f"**{source_name}** — {len(df):,} rows loaded")
    st.write("Columns:", ", ".join(map(str, df.columns)))

    user_col = find_column(
        df, ["user", "username", "userid", "user_id", "employee"]
    )
    time_col = find_column(
        df, ["date", "time", "timestamp", "datetime"]
    )

    if user_col is None:
        st.warning(
            f"{source_name}: no recognizable user column. "
            "This file will not be used for user-level scoring."
        )
        return None

    result = pd.DataFrame()
    result["user"] = df[user_col].fillna("unknown").astype(str)

    if time_col:
        result["timestamp"] = pd.to_datetime(
            df[time_col], errors="coerce"
        )
    else:
        result["timestamp"] = pd.NaT

    result["source"] = source_name
    result["hour"] = result["timestamp"].dt.hour
    result["weekday"] = result["timestamp"].dt.dayofweek

    result["after_hours"] = result["hour"].isin(
        [0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23]
    ).astype(int)

    result["weekend"] = result["weekday"].isin([5, 6]).astype(int)

    return result


if st.button("Analyze CERT Activity", type="primary"):
    activities = []

    for uploaded, name in [
        (logon_file, "logon"),
        (device_file, "device"),
        (file_file, "file")
    ]:
        if uploaded is not None:
            try:
                data = load_activity(uploaded, name)
                if data is not None:
                    activities.append(data)
            except Exception as error:
                st.error(f"Could not process {name}.csv: {error}")

    if not activities:
        st.error(
            "Upload a compatible logon.csv or device.csv file first."
        )
        st.stop()

    events = pd.concat(activities, ignore_index=True)

    features = events.groupby("user").agg(
        total_events=("source", "size"),
        logon_events=(
            "source", lambda x: (x == "logon").sum()
        ),
        device_events=(
            "source", lambda x: (x == "device").sum()
        ),
        file_events=(
            "source", lambda x: (x == "file").sum()
        ),
        after_hours_events=("after_hours", "sum"),
        weekend_events=("weekend", "sum")
    ).reset_index()

    features["after_hours_ratio"] = (
        features["after_hours_events"]
        / features["total_events"].clip(lower=1)
    )

    features["weekend_ratio"] = (
        features["weekend_events"]
        / features["total_events"].clip(lower=1)
    )

    numeric_columns = [
        c for c in features.columns
        if c != "user"
    ]

    if len(features) < 3:
        st.warning(
            "At least three distinct user profiles are needed "
            "for this initial anomaly-detection analysis."
        )
        st.dataframe(features, use_container_width=True)
        st.stop()

    X = features[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=150,
        contamination="auto",
        random_state=42
    )

    predictions = model.fit_predict(X_scaled)
    raw_scores = -model.decision_function(X_scaled)

    low = raw_scores.min()
    high = raw_scores.max()

    if high > low:
        scores = (raw_scores - low) / (high - low) * 100
    else:
        scores = np.zeros(len(raw_scores))

    features["anomaly_score"] = np.round(scores, 1)
    features["anomaly_flag"] = predictions == -1

    features["risk_level"] = np.where(
        features["anomaly_score"] >= 70, "High",
        np.where(
            features["anomaly_score"] >= 40, "Medium", "Low"
        )
    )

    features = features.sort_values(
        "anomaly_score", ascending=False
    )

    st.success("Analysis completed.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Users analyzed", len(features))
    c2.metric(
        "High-risk profiles",
        int((features["risk_level"] == "High").sum())
    )
    c3.metric(
        "ML anomalies",
        int(features["anomaly_flag"].sum())
    )

    st.subheader("Prioritized User Profiles")
    st.caption(
        "Scores are relative to this dataset, not probabilities "
        "of malicious activity."
    )
    st.dataframe(
        features, use_container_width=True, hide_index=True
    )

    st.download_button(
        "Download AEGIS Report",
        features.to_csv(index=False).encode("utf-8"),
        file_name="aegis_report.csv",
        mime="text/csv"
    )

    st.warning(
        "This initial version does not yet evaluate predictions "
        "against CERT ground-truth labels. Validate the CSV schema "
        "and investigate alerts before drawing conclusions."
    )
