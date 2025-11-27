import os
import cv2
from .detector import detect_cards
from .recognizer import Recognizer

def normalize(s):
    return s.strip().lower()

def parse_expected(filename):
    name = os.path.splitext(filename)[0]
    name = name.replace("detrebol", "de-trebol")
    parts = name.split("-")
    if "de" in parts:
        try:
            i = parts.index("de")
            rank = normalize("-".join(parts[:i]))
            suit = normalize(parts[i+1])
            color = normalize(parts[i+2]) if i+2 < len(parts) else ""
        except Exception:
            rank, suit, color = "", "", ""
    else:
        if len(parts) >= 3:
            rank, suit, color = normalize(parts[0]), normalize(parts[1]), normalize(parts[2])
        else:
            rank, suit, color = "", "", ""
    if rank == "a" or rank == "as":
        rank = "as"
    elif rank in ["j", "q", "k"]:
        rank = rank
    return rank, suit, color

def label_expected(rank, suit, color):
    if color:
        return f"{rank} de {suit} {color}"
    return f"{rank} de {suit}"

def evaluate_one(path, rec):
    img = cv2.imread(path)
    cards = detect_cards(img)
    if not cards:
        return None, None, None, None
    rank, suit, rs, ss = rec.recognize(cards[0]["warp"])
    return rank, suit, rs, ss

def main():
    root = "cartas"
    rec = Recognizer(os.path.join("assets", "templates"), cards_root=root)
    rows = []
    for fname in sorted(os.listdir(root)):
        if not fname.lower().endswith(".png"):
            continue
        exp_r, exp_s, exp_c = parse_expected(fname)
        r, s, rs, ss = evaluate_one(os.path.join(root, fname), rec)
        lbl = rec.label_es(r, s) if r and s else ""
        exp_lbl = label_expected(exp_r, exp_s, exp_c)
        ok_rank = normalize(exp_r) == normalize(rec.label_es(r, s).split(" de ")[0]) if r else False
        ok_suit = normalize(exp_s) in normalize(lbl)
        ok_color = normalize(exp_c) in normalize(lbl)
        rows.append((fname, exp_lbl, lbl, ok_rank, ok_suit, ok_color))
    out_csv = "cartas_report.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("filename,expected,recognized,rank_ok,suit_ok,color_ok\n")
        for r in rows:
            f.write(",".join([str(x) for x in r]) + "\n")
    total = len(rows)
    ok_all = sum(1 for _, _, _, rk, sk, ck in rows if rk and sk and ck)
    summary = f"total={total}, correct={ok_all}, accuracy={(ok_all/total*100.0 if total else 0):.1f}%\n"
    with open("cartas_report.txt", "w", encoding="utf-8") as f:
        f.write(summary)

if __name__ == "__main__":
    main()
