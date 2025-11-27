import os
import cv2
from .detector import detect_cards
from .recognizer import Recognizer, binarize

def extract_rank_from_image(img_path):
    img = cv2.imread(img_path)
    cards = detect_cards(img)
    if not cards:
        return None
    warp = cards[0]["warp"]
    rec = Recognizer(os.path.join("assets", "templates"))
    corner = rec.extract_corner(warp)
    rimg, _ = rec.split_rank_suit(corner)
    rb = binarize(rimg)
    return rb

def main():
    root = os.path.join("assets", "templates", "ranks")
    os.makedirs(root, exist_ok=True)
    samples = {
        "A": os.path.join("cartas", "as-de-corazones-rojo.png"),
        "2": os.path.join("cartas", "2-de-corazones-rojo.png"),
        "3": os.path.join("cartas", "3-de-corazones-rojo.png"),
        "4": os.path.join("cartas", "4-de-corazones-rojo.png"),
        "5": os.path.join("cartas", "5-de-corazones-rojo.png"),
        "6": os.path.join("cartas", "6-de-corazones-rojo.png"),
        "7": os.path.join("cartas", "7-de-corazones-rojo.png"),
        "8": os.path.join("cartas", "8-de-corazones-rojo.png"),
        "9": os.path.join("cartas", "9-de-corazones-rojo.png"),
        "10": os.path.join("cartas", "10-de-corazones-rojo.png"),
        "J": os.path.join("cartas", "J-de-corazones-rojo.png"),
        "Q": os.path.join("cartas", "Q-de-diamantes-roja.png"),
        "K": os.path.join("cartas", "K-de-picas-negra.png"),
    }
    for key, path in samples.items():
        if not os.path.exists(path):
            continue
        rank = extract_rank_from_image(path)
        if rank is None:
            continue
        outp = os.path.join(root, f"{key}.png")
        cv2.imwrite(outp, rank)

if __name__ == "__main__":
    main()

