import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# Feature columns expected after feature extraction
FEATURE_COLUMNS = [
    "block_number",
    "stride_blocks",
    "reuse_distance",
    "recent_unique_blocks",
    "recent_same_block",
]

def load_labeled_data(labeled_dir="..\\labeled"):
    # Load all CSV files produced by cache_labels.py
    csv_files = glob.glob(os.path.join(labeled_dir, "*.csv"))
    dfs = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No labeled CSV files found in {}".format(labeled_dir))
    data = pd.concat(dfs, ignore_index=True)
    return data

def ensure_features(df):
    # If feature columns are missing, compute them using the feature extractor.
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        # Import from local module – script lives in src, so relative import works.
        from features import extract_features
        df = extract_features(df)
    return df

def split_by_trace(df, test_size=0.2, random_state=42):
    # Simple random split – in a real experiment prefer trace‑wise split.
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_and_evaluate():
    data = load_labeled_data()
    data = ensure_features(data)
    X_train, X_test, y_train, y_test = split_by_trace(data)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1‑score:", f1)
    print("Confusion Matrix:\n", cm)

    # Feature importance plot
    importances = clf.feature_importances_
    plt.figure(figsize=(8, 4))
    plt.barh(FEATURE_COLUMNS, importances)
    plt.xlabel("Importance")
    plt.title("Feature Importances")
    plt.tight_layout()
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))
    os.makedirs(reports_dir, exist_ok=True)
    plt.savefig(os.path.join(reports_dir, "feature_importance.png"))
    plt.close()

    # Save the trained model
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(models_dir, "rf_classifier.joblib"))
    print("Model saved to models/rf_classifier.joblib")

if __name__ == "__main__":
    train_and_evaluate()
