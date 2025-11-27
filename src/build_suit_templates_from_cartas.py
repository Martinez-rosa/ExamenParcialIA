import os
import cv2
from .detector import detect_cards
from .recognizer import Recognizer, binarize

def extract_suit_from_image(img_path):
    img = cv2.imread(img_path)
    cards = detect_cards(img)
    if not cards:
        return None
    warp = cards[0]["warp"]
    rec = Recognizer(os.path.join("assets", "templates"))
    corner = rec.extract_corner(warp)
    rimg, simg = rec.split_rank_suit(corner)
    sb = binarize(simg)
    return sb

def main():
    root = os.path.join("assets", "templates", "suits")
    os.makedirs(root, exist_ok=True)
    pairs = [
        ("corazones", os.path.join("cartas", "as-de-corazones-rojo.png")),
        ("diamantes", os.path.join("cartas", "as-de-diamantes-roja.png")),
        ("picas", os.path.join("cartas", "as-de-picas-negra.png")),
        ("trebol", os.path.join("cartas", "as-de-trebol-negro.png")),
    ]
    for name, ip in pairs:
        if not os.path.exists(ip):
            continue
        suit = extract_suit_from_image(ip)
        if suit is None:
            continue
        outp = os.path.join(root, f"{name}.png")
        cv2.imwrite(outp, suit)

if __name__ == "__main__":
    main()

