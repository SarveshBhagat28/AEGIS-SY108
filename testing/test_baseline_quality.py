
import pandas as pd

df = pd.read_csv("data/unified_activity_features.csv")

print("DATASET SHAPE:", df.shape)

print("\nMISSING ACTIVITY FLAGS")
print(df["has_logon_activity"].isna().sum())

print("\nACTIVITY FLAG VALUES")
print(df["has_logon_activity"].value_counts(dropna=False))

print("\nCOMPUTER DATA")
print("Missing:", df["computers_used"].isna().sum())
print("Empty:", (
    df["computers_used"].fillna("").str.strip() == ""
).sum())

print("\nCOMPUTER DATA WITHOUT LOGIN ACTIVITY")

without_login = df[
    (df["has_logon_activity"] == False)
    & (df["computers_used"].notna())
    & (df["computers_used"].str.strip() != "")
]

print("Records:", len(without_login))
print(without_login[
    ["user", "calendar_date", "has_logon_activity", "computers_used"]
].head(10).to_string(index=False))
