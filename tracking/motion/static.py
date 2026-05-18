import numpy as np

# Ablation Test for no motion-help IoU matching
class NoMotion:
    def __init__(self, box):
        self.box = np.asarray(box, dtype=np.float32)

    def predict(self):
        return self.box.copy()

    def update(self, box):
        self.box = np.asarray(box, dtype=np.float32)
        return self.box.copy()