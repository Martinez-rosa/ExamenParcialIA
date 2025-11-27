import os
import cv2
import numpy as np
from .utils import auto_enhance
from .card_db import CardDB

def binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def load_templates(dir_path):
    templates = {}
    for name in sorted(os.listdir(dir_path)):
        p = os.path.join(dir_path, name)
        if not os.path.isfile(p):
            continue
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        key = os.path.splitext(name)[0]
        templates[key] = th
    return templates

def match_score(region, template, shape_weight=0.3):
    region_resized = cv2.resize(region, (template.shape[1], template.shape[0]))
    rbin = binarize(region_resized)
    res = cv2.matchTemplate(rbin, template, cv2.TM_CCOEFF_NORMED)
    max_val = float(res.max())
    rcnts, _ = cv2.findContours(rbin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tcnts, _ = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rmask = np.zeros_like(rbin)
    tmask = np.zeros_like(template)
    cv2.drawContours(rmask, rcnts, -1, 255, -1)
    cv2.drawContours(tmask, tcnts, -1, 255, -1)
    rmask = cv2.resize(rmask, (template.shape[1], template.shape[0]))
    shape_sim = 1.0
    if len(rcnts) > 0 and len(tcnts) > 0:
        shape_sim = 1.0 / (1.0 + cv2.matchShapes(rmask, tmask, cv2.CONTOURS_MATCH_I3, 0.0))
    cw = 1.0 - shape_weight
    return cw * max_val + shape_weight * shape_sim

def largest_contour(bin_img):
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

def hu_score(c1, c2):
    if c1 is None or c2 is None:
        return 0.0
    h1 = cv2.HuMoments(cv2.moments(c1)).flatten()
    h2 = cv2.HuMoments(cv2.moments(c2)).flatten()
    h1 = np.log(np.abs(h1) + 1e-9)
    h2 = np.log(np.abs(h2) + 1e-9)
    d = float(np.linalg.norm(h1 - h2))
    return 1.0 / (1.0 + d)

class Recognizer:
    def __init__(self, templates_root, cards_root=None):
        self.rank_templates = load_templates(os.path.join(templates_root, "ranks"))
        self.suit_templates = load_templates(os.path.join(templates_root, "suits"))
        self.card_db = CardDB(cards_root) if cards_root else None
        self.suit_tcontours = {}
        for k, tmpl in self.suit_templates.items():
            self.suit_tcontours[k] = largest_contour(tmpl)
        self.suit_es = {"hearts": "corazones", "diamantes": "diamantes", "clubs": "trebol", "spades": "picas", "corazones": "corazones", "diamantes": "diamantes", "trebol": "trebol", "picas": "picas"}
        self.suit_color = {"hearts": "rojo", "diamonds": "roja", "clubs": "negro", "spades": "negra", "corazones": "rojo", "diamantes": "roja", "trebol": "negro", "picas": "negra"}
        self.rank_min = 0.45
        self.suit_min = 0.5
        self.margin = 0.05
        self.last_fallback_img = None
        self.last_rank_key = None
        self.last_suit_key = None
        self.cards_min = 0.60

    def best_orientation(self, warp):
        best_w = warp
        best_s = -1.0
        best_corner = None
        for k in range(4):
            w = np.rot90(warp, k)
            c = self.extract_corner(w)
            rimg, _ = self.split_rank_suit(c)
            score = -1.0
            for tmpl in self.rank_templates.values():
                s = match_score(rimg, tmpl, shape_weight=0.3)
                if s > score:
                    score = s
            if score > best_s:
                best_s = score
                best_w = w
                best_corner = c
        return best_w, best_corner

    def extract_corner(self, card_warp, frac_h=0.28):
        h, w = card_warp.shape[:2]
        x0, y0 = int(0.02 * w), int(0.02 * h)
        x1, y1 = int(0.30 * w), int(frac_h * h)
        corner = card_warp[y0:y1, x0:x1]
        return corner

    def split_rank_suit(self, corner):
        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        inv = 255 - th
        cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 50:
                continue
            boxes.append((y, x, w, h))
        boxes.sort()
        h0, w0 = th.shape
        if len(boxes) >= 2:
            top = boxes[0]
            bottom = boxes[-1]
            ty, tx, tw, thh = top
            by, bx, bw, bh = bottom
            rank = th[ty:ty+thh, tx:tx+tw]
            suit = th[by:by+bh, bx:bx+bw]
            return rank, suit
        h = th.shape[0]
        top = th[:h//2, :]
        bottom = th[h//2:, :]
        return top, bottom

    def recognize(self, card_warp):
        oriented, corner = self.best_orientation(card_warp)
        if corner is None:
            corner = self.extract_corner(oriented, frac_h=0.28)
        hsvc = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
        vmean = float(hsvc[...,2].mean())
        gamma = 1.2 if vmean > 180 else (0.9 if vmean < 80 else None)
        corner = auto_enhance(corner, clip=2.0, gamma=gamma)
        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        h = th.shape[0]
        rroi = th[:h//2, :]
        sroi = th[h//2:, :]
        suit_color_roi = corner[h//2:, :]
        rinv = 255 - rroi
        sinv = 255 - sroi
        rcnts, _ = cv2.findContours(rinv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scnts, _ = cv2.findContours(sinv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(scnts) == 0:
            corner = self.extract_corner(oriented, frac_h=0.35)
            corner = auto_enhance(corner, clip=2.0, gamma=gamma)
            gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            h = th.shape[0]
            rroi = th[:h//2, :]
            sroi = th[h//2:, :]
            suit_color_roi = corner[h//2:, :]
            rinv = 255 - rroi
            sinv = 255 - sroi
            rcnts, _ = cv2.findContours(rinv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            scnts, _ = cv2.findContours(sinv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rbest_k, rbest_s, rbest_m2 = None, -1.0, -1.0
        rbest_img = None
        for c in rcnts:
            x, y, w, h2 = cv2.boundingRect(c)
            if w*h2 < 40:
                continue
            roi = rroi[y:y+h2, x:x+w]
            scores = []
            for k, tmpl in self.rank_templates.items():
                scores.append((k, match_score(roi, tmpl, shape_weight=0.3)))
            scores.sort(key=lambda t: t[1], reverse=True)
            if scores:
                if scores[0][1] > rbest_s:
                    rbest_k, rbest_s = scores[0]
                    rbest_m2 = scores[1][1] if len(scores) > 1 else -1.0
                    rbest_img = roi
        sbest_k, sbest_s, sbest_m2 = None, -1.0, -1.0
        sbest_img = None
        for c in scnts:
            x, y, w, h2 = cv2.boundingRect(c)
            if w*h2 < 40:
                continue
            roi = sroi[y:y+h2, x:x+w]
            scores = []
            for k, tmpl in self.suit_templates.items():
                scores.append((k, match_score(roi, tmpl, shape_weight=0.7)))
            scores.sort(key=lambda t: t[1], reverse=True)
            if scores:
                if scores[0][1] > sbest_s:
                    sbest_k, sbest_s = scores[0]
                    sbest_m2 = scores[1][1] if len(scores) > 1 else -1.0
                    sbest_img = roi
        hsv = cv2.cvtColor(suit_color_roi, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 50, 50), (179, 255, 255))
        red_ratio = (np.count_nonzero(red1) + np.count_nonzero(red2)) / float(hsv.shape[0] * hsv.shape[1])
        if red_ratio > 0.03:
            group = [k for k in self.suit_templates.keys() if k in ("hearts", "diamonds", "corazones", "diamantes")]
        else:
            group = [k for k in self.suit_templates.keys() if k in ("spades", "clubs", "picas", "trebol")]
        if group:
            gbest_k, gbest_s = None, -1.0
            broi = sbest_img if sbest_img is not None else sroi
            bbin = binarize(broi)
            bc = largest_contour(bbin)
            conv_bonus = 0.0
            is_conv = False
            if bc is not None:
                is_conv = cv2.isContourConvex(bc)
            for k in group:
                tc = self.suit_tcontours.get(k)
                if tc is None:
                    continue
                ms = 1.0 / (1.0 + cv2.matchShapes(bc, tc, cv2.CONTOURS_MATCH_I1, 0.0))
                hs = hu_score(bc, tc)
                s = 0.7 * hs + 0.3 * ms
                if is_conv and k in ("diamonds", "diamantes"):
                    s += 0.08
                if (not is_conv) and k in ("hearts", "corazones"):
                    s += 0.08
                if s > gbest_s:
                    gbest_k, gbest_s = k, s
            if gbest_s > sbest_s:
                sbest_k, sbest_s = gbest_k, gbest_s
        rank = rbest_k if rbest_s >= self.rank_min and (rbest_s - rbest_m2) >= self.margin else None
        suit = sbest_k if sbest_s >= self.suit_min and (sbest_s - sbest_m2) >= self.margin else None
        if rank is None or suit is None:
            rimg, simg = self.split_rank_suit(corner)
            rscore_best, rlabel = -1.0, None
            for k, tmpl in self.rank_templates.items():
                s = match_score(rimg, tmpl, shape_weight=0.3)
                if s > rscore_best:
                    rscore_best, rlabel = s, k
            sscore_best, slabel = -1.0, None
            for k, tmpl in self.suit_templates.items():
                s = match_score(simg, tmpl, shape_weight=0.7)
                if s > sscore_best:
                    sscore_best, slabel = s, k
            if rank is None and rscore_best >= self.rank_min:
                rank = rlabel
                rbest_s = rscore_best
            if suit is None and sscore_best >= self.suit_min:
                suit = slabel
                sbest_s = sscore_best
        used_fallback = False
        if self.card_db:
            rr = self.card_db.match(oriented)
            if rr:
                cr, cs, cc, cv, simg = rr
                if cv >= self.cards_min:
                    rank = cr.upper() if cr in ["j","q","k","as"] else ("A" if cr == "as" else cr)
                    suit = cs
                    self.last_fallback_img = simg
                    used_fallback = True
        self.last_rank_key = rank if rank is not None else rbest_k
        self.last_suit_key = suit if suit is not None else sbest_k
        return rank, suit, float(rbest_s), float(sbest_s)

    def reference_image(self, warp, mode="auto"):
        h, w = warp.shape[:2]
        if mode == "cards" and self.card_db:
            rr = self.card_db.match(warp)
            if rr:
                _, _, _, _, simg = rr
                if simg is not None:
                    return cv2.resize(simg, (w, h))
        if self.last_fallback_img is not None:
            ref = cv2.resize(self.last_fallback_img, (w, h))
            return ref
        rkey = self.last_rank_key
        skey = self.last_suit_key
        canvas = np.full_like(warp, 255)
        rt = self.rank_templates.get(rkey)
        st = self.suit_templates.get(skey)
        if rt is not None:
            rh, rw = rt.shape[:2]
            rti = cv2.resize(rt, (int(rw*0.7), int(rh*0.7)))
            rinv = cv2.bitwise_not(rti)
            y0, x0 = 10, 10
            h0, w0 = rinv.shape
            canvas[y0:y0+h0, x0:x0+w0] = cv2.cvtColor(rinv, cv2.COLOR_GRAY2BGR)
        if st is not None:
            sh, sw = st.shape[:2]
            sti = cv2.resize(st, (int(sw*0.7), int(sh*0.7)))
            sinv = cv2.bitwise_not(sti)
            y1, x1 = 10 + 60, 10
            h1, w1 = sinv.shape
            canvas[y1:y1+h1, x1:x1+w1] = cv2.cvtColor(sinv, cv2.COLOR_GRAY2BGR)
        return canvas

    def label_es(self, rank, suit):
        if rank is None or suit is None:
            return "desconocido"
        r = rank
        if r == "A":
            r = "as"
        suit_es = self.suit_es.get(suit, suit)
        color = self.suit_color.get(suit, None)
        if color:
            return f"{r} de {suit_es} {color}"
        return f"{r} de {suit_es}"

