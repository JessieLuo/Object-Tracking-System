import time
import cv2

class FpsMeter:
    def __init__(self, avg_len: int = 30):
        self.t0 = None
        self.avg_len = avg_len
        self.dts = []

    def tick(self):
        now = time.time()
        if self.t0 is None:
            self.t0 = now
            return None
        dt = now - self.t0
        self.t0 = now
        self.dts.append(dt)
        if len(self.dts) > self.avg_len:
            self.dts.pop(0)
        if len(self.dts) < 3:
            return None
        mean = sum(self.dts) / len(self.dts)
        return 1.0 / mean if mean > 0 else None


def draw_fps(vis, fps, pos=(10, 10), scale=0.6, color=(0, 255, 0)):
    text = f"FPS: {fps:.1f}"

    (w, h), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        1
    )

    x, y = pos

    # 背景
    cv2.rectangle(
        vis,
        (x - 2, y - 2),
        (x + w + 4, y + h + 4),
        (0, 0, 0),
        -1
    )

    # 文字
    cv2.putText(
        vis,
        text,
        (x, y + h),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA
    )

    return vis
