# src/step2_tracklet.py

from pathlib import Path
from collections import defaultdict, deque
import numpy as np

from pipeline.step1_detect_track import run_detect_track


# ================= CONFIG =================

TRACKLET_LEN = 16

# YOLOv8 pose index được GIỮ LẠI (chưa remap ở STEP 2)
# STEP 3 mới xử lý semantic joints
KEYPOINT_IDX = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# ---------- FILTER / FLAG THRESHOLDS ----------
EDGE_MARGIN = 40
AREA_VAR_THRESH = 0.02
LOW_MOTION_THRESH = 2.0
ID_SWITCH_DIST = 80.0
KP_MISS_RATIO = 0.35
CAMERA_MOTION_RATIO = 0.6
# ---------------------------------------------

# buffer tracklet theo track_id
buffers = defaultdict(lambda: deque(maxlen=TRACKLET_LEN))

# ================= HELPER =================

def bbox_center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], dtype=np.float32)

def bbox_area(b):
    return max((b[2] - b[0]) * (b[3] - b[1]), 1.0)

def is_near_edge(b, w, h):
    cx, cy = bbox_center(b)
    return (
        cx < EDGE_MARGIN or cx > (w - EDGE_MARGIN) or
        cy < EDGE_MARGIN or cy > (h - EDGE_MARGIN)
    )

# ================= FLAG LOGIC =================

def flag_reflection(tracklet, w, h):
    bboxes = np.array([t["bbox"] for t in tracklet], dtype=np.float32)
    centers = np.array([bbox_center(b) for b in bboxes])
    areas = np.array([bbox_area(b) for b in bboxes])

    near_edge_ratio = np.mean([is_near_edge(b, w, h) for b in bboxes])
    area_var = np.std(areas) / (np.mean(areas) + 1e-6)
    motion = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    low_motion = np.mean(motion) < LOW_MOTION_THRESH

    score = 0
    if near_edge_ratio > 0.6:
        score += 1
    if area_var < AREA_VAR_THRESH:
        score += 1
    if low_motion:
        score += 1

    return score >= 2


def flag_occlusion(tracklet):
    missing = []
    for t in tracklet:
        kp = t["keypoints"]
        if kp is None:
            missing.append(1)
        else:
            missing.append(np.isnan(kp).any())
    return np.mean(missing) > KP_MISS_RATIO


def flag_id_switch(tracklet):
    centers = np.array([bbox_center(t["bbox"]) for t in tracklet])
    jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    return np.max(jumps) > ID_SWITCH_DIST


def flag_shadow_like(tracklet):
    bboxes = np.array([t["bbox"] for t in tracklet], dtype=np.float32)
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    thin_ratio = np.mean((widths / (heights + 1e-6)) < 0.35)
    return thin_ratio > 0.6


def flag_camera_motion(all_centers):
    """
    all_centers: dict {track_id: deque([center_t-1, center_t])}
    Nếu nhiều object cùng hướng chuyển động → camera motion
    """
    if len(all_centers) < 2:
        return False

    motions = []
    for centers in all_centers.values():
        if len(centers) == 2:
            motions.append(centers[1] - centers[0])

    if len(motions) < 2:
        return False

    motions = np.array(motions)
    norms = np.linalg.norm(motions, axis=1) + 1e-6
    motions_n = motions / norms[:, None]

    sim = np.dot(motions_n, motions_n.T)
    ratio = np.mean(sim > 0.9)

    return ratio > CAMERA_MOTION_RATIO

# ================= MAIN =================

def process_tracklets(
    video_path: Path,
    model_path: Path,
    show: bool = False
):
    """
    STEP 2:
    - Gom tracklet theo track_id
    - Gắn cờ rủi ro (KHÔNG loại bỏ)
    - Yield dict cho STEP 3
    """

    frame_idx = 0
    frame_w, frame_h = None, None

    # phục vụ camera motion (frame-level)
    recent_centers = defaultdict(lambda: deque(maxlen=2))

    for r in run_detect_track(video_path, model_path, show=show):
        frame_idx += 1

        if frame_w is None:
            frame_h, frame_w = r.orig_img.shape[:2]

        if r.boxes.id is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        track_ids = r.boxes.id.cpu().numpy().astype(np.int32)

        if r.keypoints is not None:
            kps = r.keypoints.xy.cpu().numpy()
            kps = kps[:, KEYPOINT_IDX, :].astype(np.float32)
        else:
            kps = None

        # cập nhật centers cho camera motion
        for i, tid in enumerate(track_ids):
            recent_centers[int(tid)].append(
                bbox_center(boxes[i])
            )

        for i, tid in enumerate(track_ids):
            buffers[int(tid)].append({
                "track_id": int(tid),
                "bbox": boxes[i],
                "keypoints": kps[i] if kps is not None else None,
                "frame_idx": frame_idx
            })

            if len(buffers[int(tid)]) == TRACKLET_LEN:
                tracklet = list(buffers[int(tid)])

                flags = {
                    "suspected_reflection": flag_reflection(tracklet, frame_w, frame_h),
                    "occlusion": flag_occlusion(tracklet),
                    "id_switch": flag_id_switch(tracklet),
                    "shadow_like": flag_shadow_like(tracklet),
                    "camera_motion": flag_camera_motion(recent_centers),
                }

                yield {
                    "tracklet": tracklet,
                    "flags": flags
                }

# ================= TEST =================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    VIDEO_PATH = BASE_DIR / "data" / "videos" / "test.mp4"
    MODEL_PATH = BASE_DIR / "yolov8n-pose.pt"

    total = 0
    flagged = defaultdict(int)

    for item in process_tracklets(VIDEO_PATH, MODEL_PATH, show=False):
        total += 1
        for k, v in item["flags"].items():
            if v:
                flagged[k] += 1

        if total % 50 == 0:
            print(f"[STEP2] total={total} | flags={dict(flagged)}")

    print(f"[DONE] total_tracklets={total}")
    print(f"[FLAGS] {dict(flagged)}")
