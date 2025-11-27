import os
import cv2
import numpy as np

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def render_text(ch, size=(120, 120)):
    img = np.ones((size[1], size[0]), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.0
    thickness = 3
    (tw, th), _ = cv2.getTextSize(ch, font, scale, thickness)
    x = (size[0] - tw) // 2
    y = (size[1] + th) // 2
    cv2.putText(img, ch, (x, y), font, scale, 0, thickness, cv2.LINE_AA)
    return img

def render_heart(size=(120, 120)):
    img = np.ones((size[1], size[0]), dtype=np.uint8) * 255
    pts = np.array([[size[0]//2, size[1]//4], [size[0]//4, size[1]//3], [size[0]//4, size[1]//2], [size[0]//2, 3*size[1]//4], [3*size[0]//4, size[1]//2], [3*size[0]//4, size[1]//3]], dtype=np.int32)
    cv2.fillPoly(img, [pts], 0)
    return img

def render_diamond(size=(120, 120)):
    img = np.ones((size[1], size[0]), dtype=np.uint8) * 255
    pts = np.array([[size[0]//2, size[1]//4], [size[0]//4, size[1]//2], [size[0]//2, 3*size[1]//4], [3*size[0]//4, size[1]//2]], dtype=np.int32)
    cv2.fillPoly(img, [pts], 0)
    return img

def render_club(size=(120, 120)):
    img = np.ones((size[1], size[0]), dtype=np.uint8) * 255
    r = size[0]//6
    centers = [(size[0]//2, size[1]//3), (size[0]//3, size[1]//2), (2*size[0]//3, size[1]//2)]
    for c in centers:
        cv2.circle(img, c, r, 0, -1)
    cv2.rectangle(img, (size[0]//2 - r//2, size[1]//2), (size[0]//2 + r//2, 3*size[1]//4), 0, -1)
    return img

def render_spade(size=(120, 120)):
    img = np.ones((size[1], size[0]), dtype=np.uint8) * 255
    pts = np.array([[size[0]//2, size[1]//4], [size[0]//4, size[1]//2], [3*size[0]//4, size[1]//2]], dtype=np.int32)
    cv2.fillPoly(img, [pts], 0)
    cv2.circle(img, (size[0]//2, size[1]//2), size[0]//6, 0, -1)
    cv2.rectangle(img, (size[0]//2 - size[0]//20, size[1]//2), (size[0]//2 + size[0]//20, 3*size[1]//4), 0, -1)
    return img

def main():
    root = os.path.join("assets", "templates")
    ranks_dir = os.path.join(root, "ranks")
    suits_dir = os.path.join(root, "suits")
    ensure_dir(ranks_dir)
    ensure_dir(suits_dir)
    ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    for r in ranks:
        img = render_text(r)
        cv2.imwrite(os.path.join(ranks_dir, f"{r}.png"), img)
    cv2.imwrite(os.path.join(suits_dir, "hearts.png"), render_heart())
    cv2.imwrite(os.path.join(suits_dir, "diamonds.png"), render_diamond())
    cv2.imwrite(os.path.join(suits_dir, "clubs.png"), render_club())
    cv2.imwrite(os.path.join(suits_dir, "spades.png"), render_spade())

if __name__ == "__main__":
    main()

