import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from asl_features import landmark_features


PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "ML_TRAIN" else SCRIPT_DIR
DATASET_PATH = PROJECT_DIR / "ML_TRAIN" / "dataset" / "asl_data.csv"
MODEL_PATH = PROJECT_DIR / "ML_DATA" / "asl_model.joblib"
META_PATH = MODEL_PATH.with_suffix(".json")

FEATURE_COLUMNS = [f"feature_{i}" for i in range(42)]
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VALID_LABELS = LETTERS + ["SPACE", "ENTER"]
RANDOM_STATE = 42
MIN_SAMPLES_WARNING = 50
GROUP_BLOCK = 5      
MAX_FOLDS = 5


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    missing = [c for c in ["label"] + FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    keep = ["label"] + FEATURE_COLUMNS + (["session"] if "session" in df.columns else [])
    df = df[keep].copy()
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["label"] + FEATURE_COLUMNS)
    df = df[df["label"].isin(VALID_LABELS)]

    n = len(df)
    df = df.drop_duplicates(subset=["label"] + FEATURE_COLUMNS).reset_index(drop=True)  
    if n != len(df):
        print(f"Removed {n - len(df)} exact duplicate rows")
    if df.empty:
        raise ValueError("No valid samples found.")

    counts = df["label"].value_counts().reindex(VALID_LABELS, fill_value=0)
    print(f"\nDataset: {DATASET_PATH}\nValid samples: {len(df)}\n\n{counts.to_string()}")
    if (counts == 0).any():
        print("\nWARNING: no samples for:", ", ".join(counts[counts == 0].index))
    low = counts[(counts > 0) & (counts < MIN_SAMPLES_WARNING)]
    if not low.empty:
        print(f"\nWARNING: fewer than {MIN_SAMPLES_WARNING} samples (collect more!):\n{low.to_string()}")
    return df


def make_groups(df):
    """Rows recorded back-to-back are near-duplicates. They must never straddle train/test,
    otherwise accuracy is wildly inflated. Use a 'session' column if you have one."""
    if "session" in df.columns:
        return (df["label"] + "_" + df["session"].astype(str)).to_numpy()
    pos = df.groupby("label").cumcount()
    return (df["label"] + "_" + (pos // GROUP_BLOCK).astype(str)).to_numpy()


def create_models():
    feats = lambda: FunctionTransformer(landmark_features, validate=False)  
    return {
        "Random Forest": make_pipeline(feats(), RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        "Extra Trees": make_pipeline(feats(), ExtraTreesClassifier(
            n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        "SVM": make_pipeline(feats(), StandardScaler(), SVC(
            kernel="rbf", C=3.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=RANDOM_STATE)),
    }


def top_confusions(y_true, y_pred, k=8):
    pairs = pd.Series(list(zip(y_true, y_pred)))
    pairs = pairs[[a != b for a, b in pairs]].value_counts().head(k)
    return [f"{a} -> {b}: {c}" for (a, b), c in pairs.items()]

def measurable(f1, acc):
    TARGET_ACCURACY = 0.90
    TARGET_MACRO_F1 = 0.90

    print(f"Accuracy: {acc:.2%}")
    print(f"Macro F1: {f1:.4f}")

    print(
        "Accuracy criterion:",
        "MET" if acc >= TARGET_ACCURACY else "NOT MET"
    )

    print(
        "Macro F1 criterion:",
        "MET" if f1 >= TARGET_MACRO_F1 else "NOT MET"
    )


def print_success_criteria(scores):
    TARGET_ACCURACY = 0.90
    TARGET_MACRO_F1 = 0.90

    print("\n" + "=" * 50)
    print("MEASURABLE SUCCESS CRITERIA")
    print("=" * 50)

    print(f"Target Accuracy: {TARGET_ACCURACY:.0%}")
    print(f"Target Macro F1: {TARGET_MACRO_F1:.2f}")

    if not scores:
        print("\nEvaluation Status: NOT AVAILABLE")
        print("Reason: Insufficient recording groups for cross-validation.")
        print("Action: Collect more varied samples across sessions.")
        return

    print("\n=== Model Evaluation ===")

    for name, result in scores.items():
        accuracy = result["accuracy"]
        macro_f1 = result["macro_f1"]

        print(f"\nModel: {name}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(
            "Accuracy Criterion:",
            "MET" if accuracy >= TARGET_ACCURACY else "NOT MET"
        )
        print(
            "Macro F1 Criterion:",
            "MET" if macro_f1 >= TARGET_MACRO_F1 else "NOT MET"
        )




def main():
    df = load_dataset()
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy()
    groups = make_groups(df)
    models = create_models()

    groups_per_class = pd.Series(groups).groupby(y).nunique()
    n_splits = int(min(MAX_FOLDS, groups_per_class.min()))
    best_name, best_f1, scores, best_pred = "Random Forest", -1.0, {}, None

    if n_splits >= 2:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        print(f"\n=== Grouped {n_splits}-fold cross-validation (no near-duplicate leakage) ===")
        for name, model in models.items():
            pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
            acc = accuracy_score(y, pred)
            f1 = f1_score(y, pred, average="macro", zero_division=0)
            measurable(f1,acc)
            scores[name] = {"accuracy": acc, "macro_f1": f1}

            
            
            if f1 > best_f1:
                best_name, best_f1, best_pred = name, f1, pred

        labels = [l for l in VALID_LABELS if l in set(y)]
        print(f"\n=== Best: {best_name} ===")
        print(classification_report(y, best_pred, labels=labels, zero_division=0))
        print("Most common mistakes:\n  " + "\n  ".join(top_confusions(y, best_pred) or ["none"]))
        print("\nConfusion matrix (rows=true):", labels)
        print(confusion_matrix(y, best_pred, labels=labels))
        if {"J", "Z"} & set(y):
            print("\nNOTE: J and Z are motion signs; one frame can't capture them reliably.")
    else:
        print("\nWARNING: too few separate recording groups per class for a reliable evaluation. "
              "Training on all data without accuracy report. Collect more varied data.")

    print(f"\n=== Final training: {best_name} on {len(X)} samples ===")
    final = models[best_name].fit(X, y)   

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    META_PATH.write_text(json.dumps({
        "model": best_name, "samples": int(len(X)), "classes": list(final.classes_),
        "cv_scores": scores, "cv_folds": n_splits,
        "note": "Pipeline includes preprocessing; requires asl_features.py importable at inference."},
        indent=2))
    print(f"Saved: {MODEL_PATH}\nClasses: {', '.join(final.classes_)}")
    print_success_criteria(scores)
    miss = set(VALID_LABELS) - set(final.classes_)
    if miss:
        print("WARNING: model did not learn:", ", ".join(sorted(miss)))


if __name__ == "__main__":
    main()