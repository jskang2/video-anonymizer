from anonymizer.config import Config
from anonymizer.pipeline import AnonymizePipeline
import numpy as np
import cv2, os

def test_smoke(tmp_path):
    # 10프레임짜리 가짜 영상 생성 (검정 화면)
    h, w = 360, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    in_path = tmp_path / "in.mp4"
    out_path = tmp_path / "out.mp4"
    vw = cv2.VideoWriter(str(in_path), fourcc, 10, (w, h))
    for _ in range(10):
        vw.write(np.zeros((h,w,3), dtype=np.uint8))
    vw.release()

    cfg = Config(input=str(in_path), output=str(out_path))
    pipe = AnonymizePipeline(cfg)
    pipe.run(cfg.input, cfg.output)

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
