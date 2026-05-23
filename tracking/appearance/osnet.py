import os
# import time

import cv2
import numpy as np
import torch
import torchreid

# 树莓派优化
torch.set_num_threads(1)
torch.set_grad_enabled(False)


class OSNetReID:
    def __init__(self, weights_path, model_name, min_h=80, device="cpu", ):
        self.device = device
        self.min_h = min_h

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"ReID weights not found: {weights_path}")

        self.model = torchreid.models.build_model(name=model_name, num_classes=1000, pretrained=False, )

        torchreid.utils.load_pretrained_weights(self.model, weights_path, )

        self.model.eval()
        self.model.to(self.device)

    def extract(self, frame, box):
        """Extract appearance feature from the box region in the frame.
        Returns:
            feat: np.ndarray of shape (feat_dim, ), normalized to unit length.
            
            Returns None if the box is invalid or too small.
        """
        x1, y1, x2, y2 = map(int, box)

        h, w = frame.shape[:2]

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        if crop.shape[0] < self.min_h:
            # invalid box
            return None

        # =====================================================
        # preprocess
        # =====================================================
        # t0 = time.time()

        # 树莓派建议从 (128, 256) 缩小到 (64,128)，以提升速度
        crop = cv2.resize(crop, (64, 128))
        # 由于 OpenCV 读取的图像是 BGR 格式，而ReID模型需要 RGB 格式，因此需要转换颜色空间
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # 归一化处理，将像素值从 [0, 255] 转换到 [0, 1]，并转换为 float32 类型
        crop = crop.astype(np.float32) / 255.0
        # 将图像numpy数组转换为PyTorch tensor 张量
        tensor = torch.from_numpy(crop)
        # 将 (H, W, C) 转换为CNN默认格式: (C, H, W), 并且让 tensor 内存连续
        # 之后加一个批次维度，变成 (B, C, H, W) -> B=1 
        tensor = (tensor.permute(2, 0, 1).contiguous().unsqueeze(0))

        # prep_ms = (time.time() - t0) * 1000

        # =====================================================
        # forward
        # =====================================================
        # t1 = time.time()

        with torch.no_grad():
            feat = self.model(tensor)

        # forward_ms = (time.time() - t1) * 1000·
        # print(
        #     f"[ReID] prep={prep_ms:.1f}ms "
        #     f"forward={forward_ms:.1f}ms"
        # )

        # =====================================================
        # normalize
        # =====================================================
        feat = feat.cpu().numpy()[0].astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm < 1e-6:
            # 避免除以零的情况
            return None

        return feat / norm
