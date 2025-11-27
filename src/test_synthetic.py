import cv2
import numpy as np
import os
from .detector import detect_cards
from .recognizer import Recognizer

def synthetic_scene():
    h, w = 720, 960
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = (0, 128, 0)
    card = np.ones((420, 300, 3), dtype=np.uint8) * 255
    corner = card.copy()
    cv2.putText(card, "", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    cv2.putText(card, "A", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(card, (80, 160), 40, (0, 0, 255), -1)
    M = np.array([[1, 0, 200], [0, 1, 100]], dtype=np.float32)
    out = cv2.warpAffine(bg, M, (w, h))
    out[100:100+420, 200:200+300] = card
    return out

def main():
    img = synthetic_scene()
    cards = detect_cards(img)
    rec = Recognizer(os.path.join("assets", "templates"))
    out = img.copy()
    for c in cards:
        rank, suit, rs, ss = rec.recognize(c["warp"])
        label = rec.label_es(rank, suit)
        x, y = c["quad"].mean(axis=0).astype(int)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite("synthetic_result.png", out)

if __name__ == "__main__":
    main()

