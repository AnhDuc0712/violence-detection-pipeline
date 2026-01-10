import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

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

    # merge feature + label
    df = df_feat.merge(df_label, on=TRACKLET_ID, how="inner")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    groups = df["video_id"]   # 👈 split theo video (ANTI-LEAK)

    print(f"[DATA] total_samples={len(df)}")
    print(f"[DATA] positive={int(y.sum())} negative={int((y == 0).sum())}")
    print(f"[DATA] total_videos={groups.nunique()}")

    return X, y, groups, df


# ================= SAMPLE WEIGHT =================

def compute_sample_weight(df):
    """
    Giảm ảnh hưởng tracklet kém chất lượng
    (không loại bỏ sample)
    """
    weight = np.ones(len(df), dtype=np.float32)

    penalty_flags = [
        "flag_occlusion",
        "flag_id_switch",
        "flag_shadow_like",
        "flag_camera_motion",
    ]

    for f in penalty_flags:
        if f in df.columns:
            weight *= np.where(df[f] == 1, 0.5, 1.0)

    return weight


# ================= TRAIN MODELS =================

def train_xgb(X_train, y_train, sample_weight):
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

    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_lgb(X_train, y_train, sample_weight):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


# ================= MAIN =================

def main(model_type="xgb"):
    BASE_DIR = Path(__file__).resolve().parent.parent
    FEATURE_CSV = BASE_DIR / "features_step3.csv"
    LABEL_CSV = BASE_DIR / "labels.csv"

    X, y, groups, df_all = load_dataset(FEATURE_CSV, LABEL_CSV)

    # ===== SPLIT THEO VIDEO (ANTI DATA LEAK) =====
    if groups.nunique() < 5:
        print("[WARN] Too few videos, training on full data")
        X_train, X_test = X, X
        y_train, y_test = y, y
        df_train = df_all
        df_test = df_all
    else:
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        df_train = df_all.iloc[train_idx]
        df_test = df_all.iloc[test_idx]

    # ===== SAMPLE WEIGHT =====
    sample_weight = compute_sample_weight(df_train)

    # ===== TRAIN =====
    if model_type == "xgb":
        model = train_xgb(X_train, y_train, sample_weight)
    elif model_type == "lgb":
        model = train_lgb(X_train, y_train, sample_weight)
    else:
        raise ValueError("model must be xgb or lgb")

    # ===== EVALUATION (TRACKLET-LEVEL) =====
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(y_test, y_pred))

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred, digits=4))

    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"\n===== ROC-AUC =====\n{auc:.4f}")
    except Exception:
        pass

    # ===== SAVE MODEL =====
    save_model(model, model_type)


# ================= RUN =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="xgb",
        choices=["xgb", "lgb"],
        help="Classifier type"
    )
    args = parser.parse_args()

    main(args.model)
