import cv2
import numpy as np

def draw_cyber_box(img, bbox, color, label, prob, is_violent):
    x1, y1, x2, y2 = map(int, bbox)

    if is_violent:
        sub_img = img[y1:y2, x1:x2]
        if sub_img.size > 0:
            color_rect = np.zeros_like(sub_img, dtype=np.uint8)
            color_rect[:] = color
            res = cv2.addWeighted(sub_img, 0.7, color_rect, 0.3, 1.0)
            img[y1:y2, x1:x2] = res

    l = int((x2 - x1) * 0.2)
    t = 3 if is_violent else 2

    cv2.line(img, (x1, y1), (x1 + l, y1), color, t)
    cv2.line(img, (x1, y1), (x1, y1 + l), color, t)
    cv2.line(img, (x2, y1), (x2 - l, y1), color, t)
    cv2.line(img, (x2, y1), (x2, y1 + l), color, t)
    cv2.line(img, (x1, y2), (x1 + l, y2), color, t)
    cv2.line(img, (x1, y2), (x1, y2 - l), color, t)
    cv2.line(img, (x2, y2), (x2 - l, y2), color, t)
    cv2.line(img, (x2, y2), (x2, y2 - l), color, t)

    label_txt = f"{label} {prob:.2f}"
    (w, h), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    cv2.rectangle(img, (x1, y1 - 25), (x1 + w + 10, y1), color, -1)
    cv2.putText(img, label_txt, (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)