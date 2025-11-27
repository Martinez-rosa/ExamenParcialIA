import os
import cv2
import numpy as np
from .detector import detect_cards

def parse_label(fname):
    name = os.path.splitext(os.path.basename(fname))[0]
    name = name.replace("detrebol", "de-trebol")
    parts = name.split("-")
    if "de" in parts:
        i = parts.index("de")
        rank = "-".join(parts[:i]).lower()
        suit = parts[i+1].lower() if i+1 < len(parts) else ""
        color = parts[i+2].lower() if i+2 < len(parts) else ""
    else:
        rank = parts[0].lower()
        suit = parts[1].lower() if len(parts) > 1 else ""
        color = parts[2].lower() if len(parts) > 2 else ""
    if rank == "a":
        rank = "as"
    return rank, suit, color

class CardDB:
    def __init__(self, root):
        self.samples = []
        self.root = root
        if root and os.path.isdir(root):
            for fn in sorted(os.listdir(root)):
                if not fn.lower().endswith(".png"):
                    continue
                path = os.path.join(root, fn)
                img = cv2.imread(path)
                if img is None:
                    continue
                cards = detect_cards(img)
                if cards:
                    warp = cards[0]["warp"]
                else:
                    warp = img
                gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
                r, s, c = parse_label(fn)
                self.samples.append((warp, gray, r, s, c))

    def match(self, warp):
        if not self.samples:
            return None
        g = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        best = (None, None, None, -1.0, None)
        for color_samp, samp, r, s, c in self.samples:
            sr = cv2.resize(samp, (g.shape[1], g.shape[0]))
            res = cv2.matchTemplate(g, sr, cv2.TM_CCOEFF_NORMED)
            v = float(res.max())
            if v > best[3]:
                best = (r, s, c, v, color_samp)
        return best
