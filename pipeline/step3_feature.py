# src/step3_features.py

import csv
import numpy as np
from pathlib import Path

from pipeline.step2_tracklet import process_tracklets

# ================= CONFIG =================

# Mapping sau khi STEP 2 đã slice KEYPOINT_IDX
# KEYPOINT_IDX = [5,6,7,8,9,10,11,12,13,14,15,16]
# → index mới (0-based)
KP = {
    "L_SHOULDER": 0,
    "R_SHOULDER": 1,
    "L_ELBOW": 2,
    "R_ELBOW": 3,
    "L_WRIST": 4,
    "R_WRIST": 5,
    "L_HIP": 6,
    "R_HIP": 7,
    "L_KNEE": 8,
    "R_KNEE": 9,
    "L_ANKLE": 10,
    "R_ANKLE": 11,
}

# ================= HELPER =================

def bbox_center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], dtype=np.float32)

def bbox_area(b):
    return max((b[2] - b[0]) * (b[3] - b[1]), 1.0)

def euclid(a, b):
    return np.linalg.norm(a - b)

def joint_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosang, -1.0, 1.0))

# ================= FEATURE EXTRACTION =================

def extract_features(item):
    tracklet = item["tracklet"]
    flags = item["flags"]

    # ===== BBOX MOTION =====
    centers = np.array([bbox_center(t["bbox"]) for t in tracklet])
    areas = np.array([bbox_area(t["bbox"]) for t in tracklet])

    v = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    velocity_mean = np.mean(v)
    velocity_max = np.max(v)

    a = np.abs(np.diff(v))
    accel_mean = np.mean(a) if len(a) > 0 else 0.0

    bbox_scale_var = np.std(areas) / (np.mean(areas) + 1e-6)

    disp = np.linalg.norm(centers[-1] - centers[0])
    path_len = np.sum(v) + 1e-6
    trajectory_curvature = path_len / (disp + 1e-6)

    # ===== POSE DYNAMICS =====
    limb_len = []
    elbow_angles = []
    pose_energy = []

    for i, t in enumerate(tracklet):
        kp = t["keypoints"]
        if kp is None or np.isnan(kp).any():
            continue

        # lấy tay trái (ổn định hơn cho hành vi đánh)
        shoulder = kp[KP["L_SHOULDER"]]
        elbow = kp[KP["L_ELBOW"]]
        wrist = kp[KP["L_WRIST"]]

        limb_len.append(euclid(wrist, shoulder))
        elbow_angles.append(joint_angle(shoulder, elbow, wrist))

        if i > 0 and tracklet[i - 1]["keypoints"] is not None:
            prev = tracklet[i - 1]["keypoints"]
            if not np.isnan(prev).any():
                pose_energy.append(np.linalg.norm(kp - prev))

    limb_amplitude = np.mean(limb_len) if limb_len else 0.0
    joint_angle_var = np.var(elbow_angles) if elbow_angles else 0.0
    pose_motion_energy = np.mean(pose_energy) if pose_energy else 0.0

    # ===== FINAL FEATURE VECTOR =====
    features = {
        # motion
        "velocity_mean": velocity_mean,
        "velocity_max": velocity_max,
        "accel_mean": accel_mean,

        # geometry
        "bbox_scale_var": bbox_scale_var,
        "trajectory_curvature": trajectory_curvature,

        # pose
        "limb_amplitude": limb_amplitude,
        "joint_angle_var": joint_angle_var,
        "pose_motion_energy": pose_motion_energy,

        # flags (context)
        "flag_reflection": int(flags["suspected_reflection"]),
        "flag_occlusion": int(flags["occlusion"]),
        "flag_id_switch": int(flags["id_switch"]),
        "flag_shadow_like": int(flags["shadow_like"]),
        "flag_camera_motion": int(flags["camera_motion"]),
    }

    # sanitize
    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0.0

    return features

# ================= MAIN =================

def run_step3_export_csv(
    video_path: Path,
    model_path: Path,
    out_csv: Path
):
    rows = []
    video_id = video_path.stem

    for idx, item in enumerate(
        process_tracklets(video_path, model_path, show=False)
    ):
        feat = extract_features(item)
        feat["tracklet_id"] = idx
        feat["video_id"] = video_id
        rows.append(feat)

    if not rows:
        print("[STEP3] No tracklets found.")
        return

    fieldnames = ["tracklet_id", "video_id"] + [
        k for k in rows[0].keys()
        if k not in ("tracklet_id", "video_id")
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[STEP3 DONE] total_samples={len(rows)}")
    print(f"[OUTPUT] {out_csv}")

# ================= TEST =================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    VIDEO_PATH = BASE_DIR / "data" / "videos" / "test.mp4"
    MODEL_PATH = BASE_DIR / "yolov8n-pose.pt"
    OUT_CSV = BASE_DIR / "features_step3.csv"

    run_step3_export_csv(VIDEO_PATH, MODEL_PATH, OUT_CSV)
