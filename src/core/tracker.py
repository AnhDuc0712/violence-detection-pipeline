import numpy as np
from collections import defaultdict
from src.utils.geometry import bbox_center, bbox_height

# =========================
# POSE SIMILARITY
# =========================
def pose_similarity(kps1, kps2, scale):
    valid = (kps1[:, 0] > 0) & (kps2[:, 0] > 0)
    if np.sum(valid) < 3:
        return 0.0

    dist = np.linalg.norm(kps1[valid, :2] - kps2[valid, :2], axis=1)
    return float(np.exp(-np.mean(dist) / (scale * 0.5)))


# =========================
# MERGE TRACKS
# =========================
def merge_tracks(raw_tracks, MAX_FRAME_GAP, MAX_PIXEL_DIST, SIMILARITY_THRESH):
    sorted_ids = sorted(raw_tracks.keys(), key=lambda x: min(raw_tracks[x]["frame"]))

    active = {}
    merged = defaultdict(lambda: {"frame": [], "bbox": [], "kp": []})

    print("   Running Re-ID & Merge...")

    for tid in sorted_ids:
        curr = raw_tracks[tid]
        sf = curr["frame"][0]
        sc = bbox_center(curr["bbox"][0])

        best_id = None
        best_score = 0.0
        to_remove = []

        for oid, info in active.items():
            gap = sf - info["end_frame"]

            if gap > MAX_FRAME_GAP:
                to_remove.append(oid)
                continue

            if gap < 0 or np.linalg.norm(sc - info["center"]) > MAX_PIXEL_DIST:
                continue

            sim = pose_similarity(info["kp"], curr["kp"][0], info["height"])

            if sim > SIMILARITY_THRESH and sim > best_score:
                best_score = sim
                best_id = oid

        for r in to_remove:
            del active[r]

        target = best_id if best_id is not None else tid

        merged[target]["frame"].extend(curr["frame"])
        merged[target]["bbox"].extend(curr["bbox"])
        merged[target]["kp"].extend(curr["kp"])

        active[target] = {
            "end_frame": curr["frame"][-1],
            "center": bbox_center(curr["bbox"][-1]),
            "kp": curr["kp"][-1],
            "height": bbox_height(curr["bbox"][-1])
        }

    return merged
# =========================
# KALMAN SMOOTH TRACK (POSE)
# =========================
def kalman_smooth_track(kps_sequence, alpha=0.65):
    """
    Làm mượt chuỗi keypoints (T, 17, 3)
    - Dùng EMA + giữ conf
    - Ổn định hơn Savgol / moving average
    
    Args:
        kps_sequence: numpy (T, 17, 3)
        alpha: độ mượt (0.6–0.8 là đẹp)
    """

    if len(kps_sequence) < 2:
        return kps_sequence

    smoothed = kps_sequence.copy()

    for t in range(1, len(kps_sequence)):
        prev = smoothed[t - 1]
        curr = kps_sequence[t]

        # chỉ smooth x,y (không đụng conf)
        smoothed[t, :, :2] = alpha * prev[:, :2] + (1 - alpha) * curr[:, :2]

        # giữ conf gốc
        smoothed[t, :, 2] = curr[:, 2]

    return smoothed