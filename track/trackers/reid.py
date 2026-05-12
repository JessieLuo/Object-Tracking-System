import os

import cv2
import numpy as np
import torch
import torchreid


class OSNetReID:
    def __init__(
        self,
        weights_path,
        model_name="osnet_x0_25",
        min_h=80,
        device="cpu",
    ):
        self.device = device
        self.min_h = min_h

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"ReID weights not found: {weights_path}")

        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=False, # if true, would load ImageNet nor specially for person
        )

        torchreid.utils.load_pretrained_weights(
            self.model,
            weights_path,
        )

        self.model.eval()
        self.model.to(self.device)

    def extract(self, frame, box):
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
            return None

        crop = cv2.resize(crop, (128, 256))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(crop).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor /= 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            feat = self.model(tensor)

        feat = feat.cpu().numpy()[0].astype(np.float32)

        norm = np.linalg.norm(feat)

        if norm < 1e-6:
            return None

        return feat / norm