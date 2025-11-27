import argparse
import cv2
import os
import numpy as np
import time
from .detector import detect_cards
from .recognizer import Recognizer
from .utils import order_points, adjust_brightness

def draw_label(img, text, quad, pos="tl"):
    if pos == "tl":
        pts = order_points(quad)
        x, y = pts[0].astype(int)
    else:
        x, y = quad.mean(axis=0).astype(int)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    bx0, by0 = x, y - th - 6
    bx1, by1 = x + tw + 6, y + 6
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (255, 255, 255), -1)
    cv2.putText(img, text, (x + 3, y), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

def draw_hud_label(img, text):
    x, y = 10, 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (x-6, y-th-6), (x+tw+6, y+6), (255, 255, 255), -1)
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

def highlight_card(img, quad):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    pts = quad.astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    dim = (img * 0.5).astype(np.uint8)
    img[:] = np.where(mask[..., None] == 255, img, dim)
    cv2.polylines(img, [pts], True, (0, 255, 255), 3, cv2.LINE_AA)

def process_image(path, templates_root, show=False, save=None, brightness=1.0, single=False, pos="tl", save_crop=None, compare_view=False, rank_thresh=None, suit_thresh=None, save_crop_dir=None, cards_root=None, ref_mode="auto"):
    img = cv2.imread(path)
    if brightness != 1.0:
        img = adjust_brightness(img, brightness)
    cards = detect_cards(img)
    rec = Recognizer(templates_root, cards_root=cards_root)
    chosen = cards
    if single and len(cards) > 0:
        chosen = [max(cards, key=lambda c: cv2.contourArea(c["contour"]))]
    for c in chosen:
        rank, suit, rs, ss = rec.recognize(c["warp"])
        if rank_thresh is not None and suit_thresh is not None and (rs < rank_thresh or ss < suit_thresh):
            label = "desconocido"
        else:
            label = rec.label_es(rank, suit)
        draw_label(img, label, c["quad"], pos=pos)
        if save_crop:
            cv2.imwrite(save_crop, c["warp"])
        if compare_view:
            left = img.copy()
            right = c["warp"].copy()
            ref = rec.reference_image(right, mode=ref_mode)
            lh, lw = left.shape[:2]
            rh, rw = right.shape[:2]
            rfh, rfw = ref.shape[:2]
            r2 = cv2.resize(right, (int(rw*lh/rh), lh))
            rf2 = cv2.resize(ref, (int(rfw*lh/rfh), lh))
            combo = cv2.hconcat([left, r2, rf2])
            cv2.imshow("comparacion", combo)
        if save_crop_dir:
            os.makedirs(save_crop_dir, exist_ok=True)
            base = rec.label_es(rank, suit)
            ts = int(time.time()*1000)
            outp = os.path.join(save_crop_dir, f"{base}_{ts}.png")
            cv2.imwrite(outp, c["warp"])
    if show:
        cv2.imshow("resultado", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite(save, img)

def _parse_hsv_triplet(s):
    if s is None:
        return None
    parts = s.split(",")
    if len(parts) != 3:
        return None
    return [int(parts[0]), int(parts[1]), int(parts[2])]

def process_camera(cam_index, templates_root, brightness=1.0, single=False, pos="tl", compare_view=False, rank_thresh=0.55, suit_thresh=0.6, save_crop_dir=None, green_low=None, green_high=None, hold_frames=5, cards_root=None, ref_mode="auto"):
    cap = cv2.VideoCapture(cam_index)
    rec = Recognizer(templates_root, cards_root=cards_root)
    buffer = []
    stable = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame2 = adjust_brightness(frame, brightness) if brightness != 1.0 else frame
        override = None
        if green_low is not None and green_high is not None:
            override = (green_low, green_high)
        cards = detect_cards(frame2, green_override=override)
        out = frame2.copy()
        chosen = cards
        if single and len(cards) > 0:
            chosen = [max(cards, key=lambda c: cv2.contourArea(c["contour"]))]
        for c in chosen:
            rank, suit, rs, ss = rec.recognize(c["warp"])
            candidate = rec.label_es(rank, suit)
            label = candidate if (rs >= rank_thresh and ss >= suit_thresh) else "desconocido"
            highlight_card(out, c["quad"])
            if candidate != "desconocido":
                buffer.append(candidate)
                if len(buffer) > hold_frames:
                    buffer.pop(0)
                votes = {}
                for b in buffer:
                    votes[b] = votes.get(b, 0) + 1
                best_label, best_count = max(votes.items(), key=lambda t: t[1])
                min_count = max(1, int(0.6 * hold_frames))
                if best_count >= min_count:
                    stable = best_label
            draw_hud_label(out, stable if stable else label)
            if save_crop_dir:
                os.makedirs(save_crop_dir, exist_ok=True)
                ts = int(time.time()*1000)
                base = label
                outp = os.path.join(save_crop_dir, f"{base}_{ts}.png")
                cv2.imwrite(outp, c["warp"])
        if compare_view and len(chosen) > 0:
            right = chosen[0]["warp"].copy()
            oh, ow = out.shape[:2]
            rh, rw = right.shape[:2]
            r2 = cv2.resize(right, (int(rw*oh/rh), oh))
            ref = rec.reference_image(right, mode=ref_mode)
            rf = cv2.resize(ref, (int(ref.shape[1]*oh/ref.shape[0]), oh))
            combo = cv2.hconcat([out, r2, rf])
            cv2.imshow("comparacion", combo)
        cv2.imshow("cartas", out)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--templates", type=str, default=os.path.join("assets", "templates"))
    parser.add_argument("--cards", type=str, default=os.path.join("cartas"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--brightness", type=float, default=None)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--pos", type=str, default="tl")
    parser.add_argument("--save_crop", type=str, default=None)
    parser.add_argument("--compare_view", action="store_true")
    parser.add_argument("--rank_thresh", type=float, default=None)
    parser.add_argument("--suit_thresh", type=float, default=None)
    parser.add_argument("--save_crop_dir", type=str, default=None)
    parser.add_argument("--green_low", type=str, default=None)
    parser.add_argument("--green_high", type=str, default=None)
    parser.add_argument("--hold_frames", type=int, default=5)
    parser.add_argument("--ref", type=str, default="auto")
    args = parser.parse_args()
    if args.input:
        ib = args.brightness if args.brightness is not None else 1.0
        rt = args.rank_thresh if args.rank_thresh is not None else None
        st = args.suit_thresh if args.suit_thresh is not None else None
        process_image(args.input, args.templates, show=args.show, save=args.save, brightness=ib, single=args.single, pos=args.pos, save_crop=args.save_crop, compare_view=args.compare_view, rank_thresh=rt, suit_thresh=st, save_crop_dir=args.save_crop_dir, cards_root=args.cards, ref_mode=args.ref)
    elif args.camera is not None:
        cb = args.brightness if args.brightness is not None else 0.8
        gl = _parse_hsv_triplet(args.green_low)
        gh = _parse_hsv_triplet(args.green_high)
        rt = args.rank_thresh if args.rank_thresh is not None else 0.5
        st = args.suit_thresh if args.suit_thresh is not None else 0.55
        process_camera(args.camera, args.templates, brightness=cb, single=True, pos="tl", compare_view=args.compare_view, save_crop_dir=args.save_crop_dir, green_low=gl, green_high=gh, hold_frames=args.hold_frames, rank_thresh=rt, suit_thresh=st, cards_root=args.cards, ref_mode=args.ref)

if __name__ == "__main__":
    main()

