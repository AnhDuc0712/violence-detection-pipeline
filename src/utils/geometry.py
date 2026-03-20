import numpy as np

def bbox_center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], dtype=np.float32)

def bbox_height(b):
    return max(b[3] - b[1], 1.0)

def euclid(a, b):
    return np.linalg.norm(a - b)

def get_angle(a, b, c):
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)

    if nba < 1e-6 or nbc < 1e-6:
        return 0.0

    cos = np.dot(ba, bc) / (nba * nbc)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)