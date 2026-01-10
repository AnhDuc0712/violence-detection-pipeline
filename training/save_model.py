# training/save_model.py
import joblib
from pathlib import Path
from training.config import MODEL_DIR

def save_model(model, model_type):
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(MODEL_DIR) / f"{model_type}_classifier.pkl"
    joblib.dump(model, out_path)
    print(f"[MODEL SAVED] {out_path}")
