from __future__ import annotations
import cv2
import numpy as np
from typing import List, Dict

COLORS = {
    "left_elbow": (0,255,0),
    "right_elbow": (0,128,255),
    "eye": (255,0,0),
}

def draw_rois(frame: np.ndarray, rois: List[Dict]) -> np.ndarray:
    img = frame.copy()
    for r in rois:
        c = COLORS.get(r.get("label",""), (255,255,255))
        if r["shape"] == "circle":
            x,y,rad = r["params"]
            cv2.circle(img, (int(x),int(y)), int(rad), c, 2)
        elif r["shape"] == "ellipse":
            cx,cy,a,b,ang = r["params"]
            cv2.ellipse(img, (int(cx),int(cy)), (int(a),int(b)), ang, 0, 360, c, 2)
    return img
