
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


DATASET_PATH = "dataset/asl_landmarks.csv"
MODEL_PATH = "ML_DATA/asl_model.joblib"

FEATURE_COLUMNS = [f"feature_{i}" for i in range(42)]


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = ["label"] + FEATURE_COLUMNS
    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    df = df[required_columns].dropna()
    df["label"] = df["label"].astype(str).str.strip().str.upper()

    df = df[df["label"].isin(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))]

    if df.empty:
        raise ValueError("No valid A-Z samples were found.")

    print("\nSamples per letter:")
    counts = df["label"].value_counts().reindex(
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        fill_value=0
    )
    print(counts.to_string())

    if (counts < 2).any():
        print(
            "\nWARNING: Some letters have fewer than 2 samples."
            "\nThe model cannot learn reliable recognition from "
            "one sample per letter."
            "\nCollect at least 50-100 varied samples per letter "
            "before relying on predictions."
        )

    X = df[FEATURE_COLUMNS].astype(float)
    y = df["label"]

    # A test split is possible only when every class has
    # at least two examples.
    if y.value_counts().min() >= 2 and len(df) >= 52:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    else:
        print(
            "\nSkipping test evaluation because there are too "
            "few examples per class."
        )
        X_train, y_train = X, y
        X_test, y_test = None, None

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    if X_test is not None:
        predictions = model.predict(X_test)
        print(
            "\nTest accuracy:",
            round(accuracy_score(y_test, predictions), 4)
        )
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print("Classes:", ", ".join(model.classes_))


if __name__ == "__main__":
    main()
