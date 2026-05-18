import cv2
import time
from detection.yolo_det import UltralyticsYoloDetector

if __name__ == "__main__":
    img_path = "detection/data/00001355.png"
    out_path = "detection/data/00001355_det.jpg"

    detector = UltralyticsYoloDetector(
        "models/yolo26n.mnn",
        conf=0.3, classes=[0]) # 只检测person类

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Failed to read image")

    t0 = time.time()
    dets = detector.inference(img_path)
    t1 = time.time()

    print("dets shape:", dets.shape)
    print(f"latency: {(t1 - t0)*1000:.2f} ms")

    # 可视化
    for x1, y1, x2, y2, score, cls_id in dets:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{int(cls_id)}:{score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)

    # 保存结果
    ok = cv2.imwrite(out_path, img)
    if not ok:
        raise RuntimeError("Failed to save image")

    print(f"[INFO] saved to {out_path}")