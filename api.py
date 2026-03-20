from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
import os
import logging

from src.pipeline.violence_pipeline import run_pipeline

app = FastAPI()

# =========================
# LOGGER
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# =========================
# LOAD MODEL 1 LẦN
# =========================
from ultralytics import YOLO
import joblib

YOLO_MODEL = YOLO("models/yolov8n-pose.pt")
xgb_data = joblib.load("models/violence_model.pkl")

LOADED_MODELS = {
    "yolo": YOLO_MODEL,
    "xgb_booster": xgb_data['booster'] if 'booster' in xgb_data else xgb_data['model'],
    "xgb_features": xgb_data['features'],
    "xgb_scaler": xgb_data.get('scaler')
}

# =========================
# API ENDPOINT
# =========================
@app.post("/detect")
async def detect_video(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())

    # ====== SAVE FILE ======
    input_path = f"temp/input_{request_id}.mp4"
    output_path = f"temp/output_{request_id}.mp4"
    json_path = f"temp/result_{request_id}.json"

    os.makedirs("temp", exist_ok=True)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ====== CONFIG ======
    cfg = {
        "paths": {
            "video_in": input_path,
            "video_out": output_path,
            "summary_json": json_path,
            "evidence_dir": f"temp/evidence_{request_id}"
        },
        "models": {
            "conf_threshold": 0.4,
            "img_size": 640
        },
        "tracking": {
            "tracklet_len": 8,
            "max_frame_gap": 10,
            "max_pixel_dist": 100,
            "similarity_thresh": 0.5,
            "stride": 2
        },
        "logic": {
            "interaction_far": 2.5,
            "peak_vel_thresh": 1.2,
            "violence_threshold": 0.6,
            "gap_allow": 10
        }
    }

    # ====== RUN PIPELINE ======
    result = run_pipeline(cfg, logger, LOADED_MODELS, request_id)

    return result