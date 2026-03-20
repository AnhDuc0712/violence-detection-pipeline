import cv2
import joblib
import xgboost as xgb
from ultralytics import YOLO

from src.pipeline.violence_pipeline import (
    scan_phase,
    clean_phase,
    inference_phase,
    render_phase
)

from src.utils.config_loader import load_config


# =========================================================
# 🚀 MAIN
# =========================================================
def main():

    # =========================
    # LOAD CONFIG
    # =========================
    cfg = load_config()

    VIDEO_PATH = cfg["paths"]["video"]
    MODEL_PATH = cfg["paths"]["model"]
    YOLO_PATH = cfg["paths"]["yolo"]
    OUTPUT_PATH = cfg["paths"]["output"]

    CONF_THRESHOLD = cfg["yolo"]["conf_threshold"]
    IMG_SIZE = cfg["yolo"]["img_size"]

    TRACKLET_LEN = cfg["tracking"]["tracklet_len"]
    STRIDE = cfg["tracking"]["stride"]
    MAX_FRAME_GAP = cfg["tracking"]["max_frame_gap"]
    MAX_PIXEL_DIST = cfg["tracking"]["max_pixel_dist"]
    SIMILARITY_THRESH = cfg["tracking"]["similarity_thresh"]

    INTERACTION_FAR = cfg["interaction"]["interaction_far"]
    PEAK_VEL_THRESH = cfg["feature"]["peak_vel_thresh"]

    VIOLENCE_THRESHOLD = cfg["model"]["violence_threshold"]

    COLORS = cfg["visual"]["colors"]

    # =========================
    # LOAD MODEL
    # =========================
    print(f"📥 Loading Model: {MODEL_PATH}")

    data = joblib.load(MODEL_PATH)

    booster = data['booster'] if 'booster' in data else data['model']
    feature_order = data['features']
    scaler = data.get('scaler')

    # =========================
    # INIT YOLO
    # =========================
    print("🚀 Init YOLO...")
    yolo = YOLO(YOLO_PATH)

    # =========================
    # OPEN VIDEO
    # =========================
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # =========================================================
    # 🚀 PIPELINE
    # =========================================================

    # ---------- PHASE 1 ----------
    print("🚀 PHASE 1: Scan")
    raw_tracks = scan_phase(
        yolo,
        cap,
        total_frames,
        CONF_THRESHOLD,
        IMG_SIZE
    )

    cap.release()

    # ---------- PHASE 2 ----------
    print("🚀 PHASE 2: Clean")
    final_tracks = clean_phase(
        raw_tracks,
        TRACKLET_LEN,
        MAX_FRAME_GAP,
        MAX_PIXEL_DIST,
        SIMILARITY_THRESH
    )

    # ---------- PHASE 3 ----------
    print("🚀 PHASE 3: Inference")
    result_map = inference_phase(
        final_tracks,
        booster,
        feature_order,
        scaler,
        STRIDE,
        TRACKLET_LEN,
        INTERACTION_FAR,
        PEAK_VEL_THRESH,
        COLORS
    )

    # ---------- PHASE 4 ----------
    print("🚀 PHASE 4: Render")

    cap = cv2.VideoCapture(VIDEO_PATH)

    render_phase(
        cap,
        result_map,
        final_tracks,
        OUTPUT_PATH,
        fps,
        total_frames,
        VIOLENCE_THRESHOLD
    )

    cap.release()

    print("✅ DONE!")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()