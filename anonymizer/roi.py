from __future__ import annotations
import cv2
import numpy as np
from typing import List, Dict, Tuple

# ROI = dict: {"shape": "circle"|"ellipse"|"rect", "params": tuple, "label": str}

def elbows_from_keypoints(kpts: np.ndarray, safety_px: int = 12) -> List[Dict]:
    rois = []
    # Expect shape (17,2)
    def safe_pt(idx):
        if idx < 0 or idx >= len(kpts):
            return None
        x, y = kpts[idx]
        return None if (np.isnan(x) or np.isnan(y)) else (float(x), float(y))

    pairs = [(7,9,"left_elbow"), (8,10,"right_elbow")]  # (elbow, wrist)
    for e_idx, w_idx, name in pairs:
        e = safe_pt(e_idx)
        w = safe_pt(w_idx)
        if e is None:
            continue
        # radius: distance to wrist (if available) * 0.5, fallback 20px
        r = 20.0
        if e and w:
            r = max(12.0, 0.5 * (((e[0]-w[0])**2 + (e[1]-w[1])**2) ** 0.5))
        r += safety_px
        rois.append({"shape": "circle", "params": (e[0], e[1], r), "label": name})
    return rois

def eyes_from_boxes(eyes: List[Tuple[int,int,int,int]], safety_px: int = 12) -> List[Dict]:
    rois = []
    for (x,y,w,h) in eyes:
        cx, cy = x + w/2, y + h/2
        a, b = (w/2 + safety_px), (h/2 + safety_px)
        rois.append({"shape": "ellipse", "params": (cx, cy, a, b, 0), "label": "eye"})
    return rois

def draw_soft_mask(shape_hw: Tuple[int,int], rois: List[Dict], feather: int = 3) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), np.uint8)
    for roi in rois:
        if roi["shape"] == "circle":
            x, y, r = roi["params"]
            cv2.circle(mask, (int(round(x)), int(round(y))), int(round(r)), 255, -1)
        elif roi["shape"] == "ellipse":
            cx, cy, a, b, ang = roi["params"]
            cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (int(round(a)), int(round(b))), ang, 0, 360, 255, -1)
        elif roi["shape"] == "rect":
            x1,y1,x2,y2 = roi["params"]
            cv2.rectangle(mask, (int(x1),int(y1)), (int(x2),int(y2)), 255, -1)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather*2+1, feather*2+1), 0)
    return mask

def apply_anonymize(frame: np.ndarray, mask: np.ndarray, style: str = "mosaic", block: int = 16) -> np.ndarray:
    # style: mosaic|gaussian|boxblur|pixelate (pixelate==mosaic)
    out = frame.copy()
    m3 = cv2.merge([mask, mask, mask])
    inv = cv2.bitwise_not(mask)
    inv3 = cv2.merge([inv, inv, inv])

    if style in ("mosaic","pixelate"):
        # downsample masked region only
        small = cv2.resize(out, (max(1,out.shape[1]//block), max(1,out.shape[0]//block)), interpolation=cv2.INTER_LINEAR)
        pix = cv2.resize(small, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
        anon = pix
    elif style == "gaussian":
        anon = cv2.GaussianBlur(out, (25,25), 0)
    elif style == "boxblur":
        anon = cv2.blur(out, (25,25))
    else:
        anon = cv2.GaussianBlur(out, (25,25), 0)

    fg = cv2.bitwise_and(anon, m3)
    bg = cv2.bitwise_and(out, inv3)
    return cv2.add(fg, bg)
