import cv2
import numpy as np
from .utils import four_point_transform

def _estimate_green_range(image):
    h, w = image.shape[:2]
    b = max(5, min(h, w) // 50)
    top = image[:b, :]
    bottom = image[-b:, :]
    left = image[:, :b]
    right = image[:, -b:]
    border = np.vstack([top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)])
    hsvb = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hmed = np.median(hsvb[:, 0])
    smed = np.median(hsvb[:, 1])
    vmed = np.median(hsvb[:, 2])
    if 30 <= hmed <= 90 and smed >= 40:
        hl = max(0, hmed - 20)
        hu = min(179, hmed + 20)
        sl = max(30, smed - 30)
        su = 255
        vl = max(30, vmed - 30)
        vu = 255
        return np.array([hl, sl, vl], dtype=np.uint8), np.array([hu, su, vu], dtype=np.uint8)
    return np.array([35, 40, 40], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8)

def detect_cards(image, green_override=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green, upper_green = _estimate_green_range(image)
    if green_override is not None:
        lg, ug = green_override
        lower_green = np.array(lg, dtype=np.uint8)
        upper_green = np.array(ug, dtype=np.uint8)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_not_green = cv2.bitwise_not(mask_green)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    wmask = cv2.inRange(L, 170, 255)
    amask = cv2.inRange(cv2.absdiff(A, np.full_like(A, 128)), 0, 35)
    bmask = cv2.inRange(cv2.absdiff(B, np.full_like(B, 128)), 0, 35)
    white_like = cv2.bitwise_and(wmask, cv2.bitwise_and(amask, bmask))
    mask = cv2.bitwise_and(mask_not_green, white_like)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = []
    h, w = image.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (w * h) * 0.01:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        box = None
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(cnt)
            bw, bh = rect[1]
            if bw == 0 or bh == 0:
                continue
            ratio = max(bw, bh) / max(1.0, min(bw, bh))
            if ratio < 1.25 or ratio > 1.75:
                continue
            box = cv2.boxPoints(rect)
            pts = box.astype(np.float32)
        warped = four_point_transform(image, pts)
        labw = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        Lw, Aw, Bw = cv2.split(labw)
        wmask_w = cv2.inRange(Lw, 170, 255)
        amask_w = cv2.inRange(cv2.absdiff(Aw, np.full_like(Aw, 128)), 0, 35)
        bmask_w = cv2.inRange(cv2.absdiff(Bw, np.full_like(Bw, 128)), 0, 35)
        white_like_w = cv2.bitwise_and(wmask_w, cv2.bitwise_and(amask_w, bmask_w))
        white_ratio = np.count_nonzero(white_like_w) / float(warped.shape[0] * warped.shape[1])
        if white_ratio < 0.35:
            continue
        rect_area = cv2.contourArea(pts.reshape(-1, 1, 2))
        if rect_area <= 0:
            continue
        solidity = area / rect_area
        if solidity < 0.5:
            continue
        cards.append({"contour": cnt, "quad": pts, "warp": warped})
    if len(cards) == 0:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k2, iterations=2)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < (w * h) * 0.005:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
            else:
                rect = cv2.minAreaRect(cnt)
                bw, bh = rect[1]
                if bw == 0 or bh == 0:
                    continue
                ratio = max(bw, bh) / max(1.0, min(bw, bh))
                if ratio < 1.2 or ratio > 1.8:
                    continue
                pts = cv2.boxPoints(rect).astype(np.float32)
            warped = four_point_transform(image, pts)
            labw = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
            Lw, Aw, Bw = cv2.split(labw)
            wmask_w = cv2.inRange(Lw, 170, 255)
            amask_w = cv2.inRange(cv2.absdiff(Aw, np.full_like(Aw, 128)), 0, 35)
            bmask_w = cv2.inRange(cv2.absdiff(Bw, np.full_like(Bw, 128)), 0, 35)
            white_like_w = cv2.bitwise_and(wmask_w, cv2.bitwise_and(amask_w, bmask_w))
            white_ratio = np.count_nonzero(white_like_w) / float(warped.shape[0] * warped.shape[1])
            if white_ratio < 0.35:
                continue
            rect_area = cv2.contourArea(pts.reshape(-1, 1, 2))
            if rect_area <= 0:
                continue
            solidity = area / rect_area
            if solidity < 0.4:
                continue
            cards.append({"contour": cnt, "quad": pts, "warp": warped})
    return cards

