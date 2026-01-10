import argparse
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from training.config import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    TRACKLET_ID,
    TEST_SIZE,
    RANDOM_STATE
)
from training.save_model import save_model


# ================= LOAD DATASET =================

def load_dataset(feature_csv: Path, label_csv: Path):
    df_feat = pd.read_csv(feature_csv)
    df_label = pd.read_csv(label_csv)

    # merge label thật
    df = df_feat.merge(df_label, on=TRACKLET_ID, how="inner")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    print(f"[DATA] total_samples={len(df)}")
    print(f"[DATA] positive={int(y.sum())} negative={int((y == 0).sum())}")

    return X, y


# ================= TRAIN MODELS =================

def train_xgb(X_train, y_train):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_lgb(X_train, y_train):
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


# ================= MAIN =================

def main(model_type="xgb"):
    BASE_DIR = Path(__file__).resolve().parent.parent
    FEATURE_CSV = BASE_DIR / "features_step3.csv"
    LABEL_CSV = BASE_DIR / "labels.csv"

    X, y = load_dataset(FEATURE_CSV, LABEL_CSV)

    # ===== SPLIT LOGIC (SAFE FOR SMALL DATASET) =====
    if len(y) < 20:
        print("[WARN] Dataset too small, training on full data (no split)")
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

    # ===== TRAIN =====
    if model_type == "xgb":
        model = train_xgb(X_train, y_train)
    elif model_type == "lgb":
        model = train_lgb(X_train, y_train)
    else:
        raise ValueError("model must be xgb or lgb")

    # ===== EVALUATION =====
    y_pred = model.predict(X_test)

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(y_test, y_pred))

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred, digits=4))

    save_model(model, model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "lgb"])
    args = parser.parse_args()

    main(args.model)
