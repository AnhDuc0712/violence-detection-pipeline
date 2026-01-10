# src/step1_detect_track.py

from ultralytics import YOLO
from pathlib import Path
from typing import Generator


def run_detect_track(
    video_path: Path,
    model_path: Path,
    show: bool = False
) -> Generator:
    """
    STEP 1:
    - Detect PERSON only
    - Track with ByteTrack
    - Extract pose keypoints
    - Stream frame-by-frame (generator)

    Yield:
        ultralytics.yolo.engine.results.Results
    """

    assert video_path.exists(), f"Video not found: {video_path}"
    assert model_path.exists(), f"Model not found: {model_path}"

    # Load YOLOv8 pose model
    model = YOLO(str(model_path))

    results = model.track(
        source=str(video_path),
        classes=[0],              # ✅ ONLY PERSON
        tracker="bytetrack.yaml",
        persist=True,             # keep track_id
        stream=True,              # generator mode
        show=show
    )

    for r in results:
        yield r


# ================= DEBUG / TEST =================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    VIDEO_PATH = BASE_DIR / "data" / "videos" / "test.mp4"
    MODEL_PATH = BASE_DIR / "yolov8n-pose.pt"

    for r in run_detect_track(VIDEO_PATH, MODEL_PATH, show=True):
        # debug only
        pass
