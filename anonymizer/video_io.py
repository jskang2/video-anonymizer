from __future__ import annotations
import cv2
from typing import Tuple

class VideoReader:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def __iter__(self):
        return self

    def __next__(self):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.release()
            raise StopIteration
        return frame

class VideoWriter:
    def __init__(self, path: str, width: int, height: int, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self.out.isOpened():
            raise RuntimeError(f"Cannot open writer: {path}")

    def write(self, frame):
        self.out.write(frame)

    def release(self):
        self.out.release()
