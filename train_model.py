import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    metrics: dict


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def train(
    csv_path: Path,
    out_path: Path,
    imbalance: str,
    random_state: int,
    n_estimators: int,
) -> TrainResult:
    df = pd.read_csv(csv_path)
    _validate_schema(df)

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    # Mandatory: 70/30 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )

    if imbalance == "smote":
        # SMOTE is applied only to the training set.
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_res, y_res)
        model = clf
    elif imbalance == "class_weight":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)
        model = clf
    else:
        raise ValueError("imbalance must be one of: smote, class_weight")

    # Metrics (probability-focused; accuracy is misleading for this dataset)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "classification_report": classification_report(y_test, pred, digits=4),
        "feature_names": FEATURES,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "fraud_rate": float(y.mean()),
        "imbalance_strategy": imbalance,
        "random_state": int(random_state),
        "n_estimators": int(n_estimators),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": metrics}, out_path)

    return TrainResult(model_path=out_path, metrics=metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("creditcard.csv"))
    parser.add_argument("--out", type=Path, default=Path("ml/model.pkl"))
    parser.add_argument("--imbalance", choices=["smote", "class_weight"], default="smote")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=400)
    args = parser.parse_args()

    res = train(
        csv_path=args.csv,
        out_path=args.out,
        imbalance=args.imbalance,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )

    print(f"Saved model to: {res.model_path}")
    print(f"ROC-AUC: {res.metrics['roc_auc']:.5f}")
    print(f"Avg Precision (PR-AUC): {res.metrics['avg_precision']:.5f}")
    print("Confusion matrix:", res.metrics["confusion_matrix"])
    print()
    print(res.metrics["classification_report"])


if __name__ == "__main__":
    main()

