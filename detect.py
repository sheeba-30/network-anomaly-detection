import sys

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

if len(sys.argv) != 2:
    raise SystemExit("Usage: python detect.py flows.csv")

data = pd.read_csv(sys.argv[1])
features = data.select_dtypes(include="number")
if features.empty or len(features) < 10:
    raise SystemExit("Provide at least 10 rows and one numeric feature column.")

detector = make_pipeline(
    SimpleImputer(strategy="median"),
    IsolationForest(contamination="auto", random_state=42, n_estimators=200),
)
labels = detector.fit_predict(features)
result = data.copy()
result["anomaly"] = labels == -1
count = int(result["anomaly"].sum())
print(f"Reviewed {len(result)} flows; flagged {count} for investigation.")
result[result["anomaly"]].to_csv("flagged_flows.csv", index=False)
print("Wrote flagged rows to flagged_flows.csv. Validate alerts with domain context.")
