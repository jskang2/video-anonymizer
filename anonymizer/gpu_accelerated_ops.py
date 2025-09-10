import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

class GPUAcceleratedOps:
    """GPU 가속 이미지 처리 연산들"""
    
    def __init__(self, device='cuda:0'):
        # Device 정규화: 이미 cuda: 포함된 경우 중복 방지
        if torch.cuda.is_available():
            if not str(device).startswith('cuda:') and device != 'cpu':
                self.device = f"cuda:{device}"
            else:
                self.device = str(device)
        else:
            self.device = 'cpu'
        print(f"[GPU Ops] Device: {self.device}")
    
    def bgr_to_gray_gpu(self, frame: np.ndarray) -> np.ndarray:
        """GPU에서 BGR → Gray 변환"""
        if self.device == 'cpu':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            # 입력 검증
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # CPU → GPU
            frame_tensor = torch.from_numpy(frame).to(self.device).float()
            frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC → CHW
            
            # BGR → Gray (GPU에서 계산)
            # Gray = 0.114*B + 0.587*G + 0.299*R
            weights = torch.tensor([0.114, 0.587, 0.299], device=self.device)
            gray_tensor = torch.sum(frame_tensor * weights.view(3, 1, 1), dim=0)
            
            # GPU → CPU
            gray = gray_tensor.cpu().numpy().astype(np.uint8)
            return gray
            
        except Exception as e:
            print(f"[GPU Gray] GPU 변환 실패, CPU 폴백: {e}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def draw_mask_gpu(self, shape_hw: Tuple[int,int], rois: List[Dict]) -> np.ndarray:
        """GPU에서 마스크 생성 (circle과 ellipse만 지원)"""
        h, w = shape_hw
        
        if not rois or self.device == 'cpu':
            # CPU 폴백
            return self._draw_mask_cpu(shape_hw, rois)
        
        try:
            # GPU에서 마스크 생성
            mask_tensor = torch.zeros((h, w), device=self.device, dtype=torch.float32)
            
            for roi in rois:
                if roi["shape"] == "circle":
                    x, y, r = roi["params"]
                    mask_tensor = self._add_circle_gpu(mask_tensor, x, y, r)
                elif roi["shape"] == "ellipse":
                    cx, cy, a, b, ang = roi["params"]
                    mask_tensor = self._add_ellipse_gpu(mask_tensor, cx, cy, a, b)
            
            # GPU → CPU
            mask = (mask_tensor * 255).cpu().numpy().astype(np.uint8)
            
            # Gaussian blur는 CPU에서 (OpenCV가 더 빠름)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            
            return mask
            
        except Exception as e:
            print(f"[GPU Mask] GPU 처리 실패, CPU 폴백: {e}")
            return self._draw_mask_cpu(shape_hw, rois)
    
    def _add_circle_gpu(self, mask_tensor: torch.Tensor, cx: float, cy: float, r: float):
        """GPU에서 원 그리기"""
        h, w = mask_tensor.shape
        
        # 좌표 그리드 생성
        y_coords = torch.arange(h, device=self.device, dtype=torch.float32).view(-1, 1)
        x_coords = torch.arange(w, device=self.device, dtype=torch.float32).view(1, -1)
        
        # 거리 계산
        dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
        circle_mask = (dist_sq <= r ** 2).float()
        
        # 마스크 추가 (OR 연산)
        mask_tensor = torch.max(mask_tensor, circle_mask)
        
        return mask_tensor
    
    def _add_ellipse_gpu(self, mask_tensor: torch.Tensor, cx: float, cy: float, a: float, b: float):
        """GPU에서 타원 그리기 (회전 없음)"""
        h, w = mask_tensor.shape
        
        # 좌표 그리드 생성
        y_coords = torch.arange(h, device=self.device, dtype=torch.float32).view(-1, 1)
        x_coords = torch.arange(w, device=self.device, dtype=torch.float32).view(1, -1)
        
        # 타원 방정식: (x-cx)²/a² + (y-cy)²/b² <= 1
        ellipse_eq = ((x_coords - cx) / a) ** 2 + ((y_coords - cy) / b) ** 2
        ellipse_mask = (ellipse_eq <= 1.0).float()
        
        # 마스크 추가
        mask_tensor = torch.max(mask_tensor, ellipse_mask)
        
        return mask_tensor
    
    def _draw_mask_cpu(self, shape_hw: Tuple[int,int], rois: List[Dict]) -> np.ndarray:
        """CPU 폴백"""
        h, w = shape_hw
        mask = np.zeros((h, w), np.uint8)
        
        for roi in rois:
            if roi["shape"] == "circle":
                x, y, r = roi["params"]
                cv2.circle(mask, (int(round(x)), int(round(y))), int(round(r)), 255, -1)
            elif roi["shape"] == "ellipse":
                cx, cy, a, b, ang = roi["params"]
                cv2.ellipse(mask, (int(round(cx)), int(round(cy))), 
                           (int(round(a)), int(round(b))), ang, 0, 360, 255, -1)
        
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        return mask
    
    def apply_anonymize_gpu(self, frame: np.ndarray, mask: np.ndarray, style: str = "mosaic") -> np.ndarray:
        """GPU 가속 익명화 처리"""
        if self.device == 'cpu':
            return self._apply_anonymize_cpu(frame, mask, style)
        
        try:
            # CPU → GPU
            frame_tensor = torch.from_numpy(frame).to(self.device).float() / 255.0
            mask_tensor = torch.from_numpy(mask).to(self.device).float() / 255.0
            
            if style in ("mosaic", "pixelate"):
                # 다운샘플링 → 업샘플링
                h, w = frame_tensor.shape[:2]
                small_h, small_w = max(1, h // 16), max(1, w // 16)
                
                # GPU에서 리사이즈
                frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # BHWC → BCHW
                small = F.interpolate(frame_tensor, size=(small_h, small_w), mode='bilinear')
                pixelated = F.interpolate(small, size=(h, w), mode='nearest')
                pixelated = pixelated.squeeze(0).permute(1, 2, 0)  # BCHW → HWC
                
                anon_tensor = pixelated
            else:  # gaussian, boxblur
                # GPU 블러는 복잡하므로 CPU 폴백
                return self._apply_anonymize_cpu(frame, mask, style)
            
            # 마스크 적용 (차원 맞추기)
            mask_3ch = mask_tensor.unsqueeze(2).expand(-1, -1, 3)  # HW → HW3
            
            # anon_tensor는 HWC, frame_tensor를 HWC로 되돌림
            frame_tensor_hwc = frame_tensor.squeeze(0).permute(1, 2, 0)  # BCHW → HWC
            
            result = frame_tensor_hwc * (1 - mask_3ch) + anon_tensor * mask_3ch
            
            # GPU → CPU
            result = (result * 255).cpu().numpy().astype(np.uint8)
            return result
            
        except Exception as e:
            print(f"[GPU Anonymize] GPU 처리 실패, CPU 폴백: {e}")
            return self._apply_anonymize_cpu(frame, mask, style)
    
    def _apply_anonymize_cpu(self, frame: np.ndarray, mask: np.ndarray, style: str) -> np.ndarray:
        """CPU 폴백"""
        out = frame.copy()
        m3 = cv2.merge([mask, mask, mask])
        inv = cv2.bitwise_not(mask)
        inv3 = cv2.merge([inv, inv, inv])

        if style in ("mosaic","pixelate"):
            small = cv2.resize(out, (max(1,out.shape[1]//16), max(1,out.shape[0]//16)), 
                             interpolation=cv2.INTER_LINEAR)
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