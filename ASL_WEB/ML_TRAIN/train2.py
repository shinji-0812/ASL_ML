
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# FILE PATHS
# ==========================================

DATASET_PATH = "ASL_WEB/ML_TRAIN/dataset/asl_landmarks.csv"
MODEL_PATH = "ASL_WEB/ML_DATA/asl_model.joblib"


# ==========================================
# SETTINGS
# ==========================================

FEATURE_COLUMNS = [
    f"feature_{i}"
    for i in range(42)
]

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
COMMANDS = ["SPACE", "ENTER"]
VALID_LABELS = LETTERS + COMMANDS


# ==========================================
# TRAIN MODEL
# ==========================================

def main():

    # Create model folder if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Check dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    # Load CSV
    df = pd.read_csv(DATASET_PATH)

    # Check required columns
    required_columns = ["label"] + FEATURE_COLUMNS

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    # Keep required columns only
    df = df[required_columns].copy()

    # Clean labels
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Convert features to numeric values
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove incomplete rows
    df = df.dropna(subset=required_columns)

    # Keep A-Z, SPACE, and ENTER
    df = df[df["label"].isin(VALID_LABELS)]

    if df.empty:
        raise ValueError(
            "No valid ASL samples were found."
        )

    # ==========================================
    # SHOW DATASET INFORMATION
    # ==========================================

    print("\n================================")
    print("SIGN SPEAK ASL MODEL TRAINING")
    print("================================")

    print(f"\nTotal samples: {len(df)}")

    print("\nSamples per label:")

    counts = (
        df["label"]
        .value_counts()
        .reindex(VALID_LABELS, fill_value=0)
    )

    print(counts.to_string())

    missing_labels = counts[counts == 0].index.tolist()

    if missing_labels:
        print("\nWARNING: These labels have no samples:")
        print(", ".join(missing_labels))

    low_sample_labels = counts[
        (counts > 0) & (counts < 50)
    ]

    if not low_sample_labels.empty:
        print(
            "\nWARNING: Some labels have fewer than "
            "50 samples:"
        )
        print(low_sample_labels.to_string())


    if (counts[counts > 0] < 2).any():
        raise ValueError(
            "Every label with samples must have at least "
            "2 samples to train reliably with a stratified split. "
            "Collect more samples for the affected labels."
        )



    X = df[FEATURE_COLUMNS].astype(float)
    y = df["label"]



    if y.value_counts().min() >= 2 and len(df) >= 52:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        evaluate_model = True

    else:
        print(
            "\nWARNING: Not enough data for a reliable "
            "test evaluation."
        )

        X_train = X
        y_train = y

        X_test = None
        y_test = None

        evaluate_model = False



    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )



    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("Training completed!")



    if evaluate_model:

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f"\nTest accuracy: {accuracy:.4f}"
        )

        print("\nClassification report:")

        print(
            classification_report(
                y_test,
                predictions,
                labels=VALID_LABELS,
                zero_division=0
            )
        )



    joblib.dump(model, MODEL_PATH)

    print("\n================================")
    print("MODEL SAVED SUCCESSFULLY")
    print("================================")

    print(f"\nModel location: {MODEL_PATH}")

    print("\nModel classes:")

    print(", ".join(model.classes_))




if __name__ == "__main__":
    main()
