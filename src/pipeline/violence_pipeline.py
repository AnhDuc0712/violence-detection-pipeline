import cv2
import numpy as np
import joblib
import xgboost as xgb
from collections import defaultdict
from tqdm import tqdm
import os
import json

from src.utils.geometry import bbox_center, compute_iou
from src.utils.visualizer import draw_cyber_box
from src.core.tracker import merge_tracks
from src.core.features import compute_features_advanced, get_torso_scale
from src.core.decision import classify_interaction_state


# =========================================================
# 🚀 PHASE 1: SCAN
# =========================================================
def scan_phase(yolo, cap, total_frames, CONF_THRESHOLD, IMG_SIZE):
    raw_tracks = defaultdict(lambda: {"frame": [], "bbox": [], "kp": []})

    for frame_idx in tqdm(range(total_frames), desc="Scanning"):
        ret, frame = cap.read()
        if not ret:
            break

        res = yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False,
            classes=[0]
        )

        if res and res[0].boxes.id is not None and res[0].keypoints is not None:
            kps_xy = res[0].keypoints.xy.cpu().numpy()

            if hasattr(res[0].keypoints, 'conf') and res[0].keypoints.conf is not None:
                kps_conf = res[0].keypoints.conf.cpu().numpy()
            else:
                kps_conf = np.ones((kps_xy.shape[0], kps_xy.shape[1]))

            for b, i, k_xy, k_cf in zip(
                res[0].boxes.xyxy.cpu().numpy(),
                res[0].boxes.id.cpu().numpy().astype(int),
                kps_xy,
                kps_conf
            ):
                raw_tracks[i]["frame"].append(frame_idx)
                raw_tracks[i]["bbox"].append(b)

                kp_full = np.concatenate([k_xy, k_cf[:, None]], axis=1)
                raw_tracks[i]["kp"].append(kp_full)

    return raw_tracks


# =========================================================
# 🧹 PHASE 2: CLEAN
# =========================================================
def clean_phase(raw_tracks, TRACKLET_LEN, MAX_FRAME_GAP, MAX_PIXEL_DIST, SIMILARITY_THRESH):
    clean_dict = merge_tracks(raw_tracks, MAX_FRAME_GAP, MAX_PIXEL_DIST, SIMILARITY_THRESH)

    final_tracks = {}

    for tid, t in clean_dict.items():
        if len(t["frame"]) < TRACKLET_LEN:
            continue

        idx = np.argsort(t["frame"])

        sorted_kps = np.array(t["kp"])[idx]

        # ⚠️ giữ nguyên logic interpolate của bạn
        smoothed_kps = sorted_kps

        avg_scale = get_torso_scale(smoothed_kps)

        final_tracks[tid] = {
            "frames": np.array(t["frame"])[idx],
            "bbox": np.array(t["bbox"])[idx],
            "kp": smoothed_kps,
            "scale_cache": avg_scale
        }

    return final_tracks


# =========================================================
# 🧠 PHASE 3: INFERENCE
# =========================================================
def inference_phase(
    final_tracks,
    booster,
    feature_order,
    scaler,
    STRIDE,
    TRACKLET_LEN,
    INTERACTION_FAR,
    PEAK_VEL_THRESH,
    COLORS
):
    frame_map = defaultdict(list)

    for tid, t in final_tracks.items():
        sc = t["scale_cache"]
        for i, f in enumerate(t["frames"]):
            frame_map[f].append({
                "tid": tid,
                "center": bbox_center(t["bbox"][i]),
                "scale": sc
            })

    result_map = defaultdict(dict)

    safety_lock = defaultdict(int)
    hugging_count = defaultdict(int)
    ema_prob = defaultdict(float)

    for tid, t in tqdm(final_tracks.items(), desc="Inference"):
        frames = t["frames"]
        kps = t["kp"]
        bboxes = t["bbox"]
        sc = t["scale_cache"]

        for i in range(0, len(frames) - TRACKLET_LEN + 1, STRIDE):
            chunk_kps = kps[i:i+TRACKLET_LEN]
            chunk_bbox = bboxes[i:i+TRACKLET_LEN]

            sf = frames[i]

            peers = frame_map[sf]
            min_dist = INTERACTION_FAR
            curr_c = bbox_center(chunk_bbox[0])

            if len(peers) > 1:
                for p in peers:
                    if p["tid"] == tid:
                        continue

                    mask = final_tracks[p["tid"]]["frames"] == sf
                    if np.any(mask):
                        other_bbox = final_tracks[p["tid"]]["bbox"][mask][0]
                    else:
                        continue

                    iou = compute_iou(chunk_bbox[0], other_bbox)

                    if iou > 0.3:
                        min_dist = 0.5
                        break
                    else:
                        d = np.linalg.norm(curr_c - p["center"]) / sc
                        min_dist = min(min_dist, d)

            feat_dict, w_vel, w_jerk = compute_features_advanced(
                chunk_kps,
                chunk_bbox,
                min_dist,
                sc,
                PEAK_VEL_THRESH
            )

            vec = np.array([feat_dict[f] for f in feature_order]).reshape(1, -1)

            if scaler:
                vec = scaler.transform(vec)

            raw_prob_curr = float(
                booster.predict(xgb.DMatrix(vec, feature_names=feature_order))[0]
            )

            raw_prob = 0.6 * ema_prob[tid] + 0.4 * raw_prob_curr
            ema_prob[tid] = raw_prob

            adj, label, color = classify_interaction_state(
                feat_dict, w_vel, w_jerk, raw_prob, COLORS
            )

            if label == "HUGGING":
                hugging_count[tid] += 1
            else:
                hugging_count[tid] = 0

            if hugging_count[tid] >= 8:
                safety_lock[tid] = 30

            if safety_lock[tid] > 0:
                final_prob = min(raw_prob, 0.35)
                safety_lock[tid] -= 1
            else:
                final_prob = np.clip(raw_prob + adj, 0.0, 1.0)

            for j in range(TRACKLET_LEN):
                curr_f = frames[i+j]

                if tid not in result_map[curr_f] or final_prob > result_map[curr_f][tid].get("prob", 0):
                    result_map[curr_f][tid] = {
                        "prob": final_prob,
                        "raw_prob": raw_prob,
                        "label": label,
                        "color": color
                    }

    return result_map


# =========================================================
# 🎥 PHASE 4: RENDER
# =========================================================
def render_phase(
    cap,
    result_map,
    final_tracks,
    OUTPUT_PATH,
    fps,
    total_frames,
    VIOLENCE_THRESHOLD
):
    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (int(cap.get(3)), int(cap.get(4)))
    )

    frame_idx = 0

    for _ in tqdm(range(total_frames), desc="Rendering"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in result_map:
            for tid, info in result_map[frame_idx].items():
                t_data = final_tracks[tid]
                mask = t_data["frames"] == frame_idx

                if np.any(mask):
                    bbox = t_data["bbox"][mask][0]

                    draw_cyber_box(
                        frame,
                        bbox,
                        info["color"],
                        info["label"],
                        info["prob"],
                        info["prob"] > VIOLENCE_THRESHOLD
                    )

        out.write(frame)
        frame_idx += 1

    out.release()
    return summary