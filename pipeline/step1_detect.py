# src/step1_detect_track.py
from ultralytics import YOLO
from pathlib import Path

def run_detect_track(
    video_path: Path,
    model_path: Path,
    show: bool = True
):
    """
    Generator trả về từng frame result của YOLO (đã track + pose)
    """
    model = YOLO(str(model_path))

    results = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        show=show
    )

    for r in results:
        yield r


# ================= TEST CHẠY RIÊNG =================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    VIDEO_PATH = BASE_DIR / "data" / "videos" / "test.mp4"
    MODEL_PATH = BASE_DIR / "yolov8n-pose.pt"

    for _ in run_detect_track(VIDEO_PATH, MODEL_PATH):
        pass
