import json
import sys
from pathlib import Path

import joblib
import pandas as pd


FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def _coerce_payload(payload: dict) -> pd.DataFrame:
    row = {}
    for col in FEATURES:
        v = payload.get(col)
        if v is None:
            # Allow missing fields; default to 0.0 for synthetic/demo transactions.
            v = 0.0
        row[col] = float(v)
    return pd.DataFrame([row], columns=FEATURES)


def main() -> None:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ml/model.pkl")
    blob = joblib.load(model_path)
    model = blob["model"]
    meta = blob.get("meta", {})

    payload = json.load(sys.stdin)
    X = _coerce_payload(payload)

    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)

    out = {
        "fraud_probability": proba,
        "prediction": pred,  # 1 = fraud, 0 = legitimate
        "threshold": 0.5,
        "model_meta": {
            "imbalance_strategy": meta.get("imbalance_strategy"),
            "roc_auc": meta.get("roc_auc"),
            "avg_precision": meta.get("avg_precision"),
        },
    }
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
