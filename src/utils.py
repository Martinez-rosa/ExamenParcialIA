import numpy as np
import cv2

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts, w=300, h=420):
    rect = order_points(pts)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

def adjust_brightness(image, scale=1.0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.float32) * float(scale), 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def auto_enhance(image, clip=2.0, gamma=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    v2 = clahe.apply(v)
    if gamma is not None:
        g = float(gamma)
        v2 = np.clip((v2 / 255.0) ** g * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

