import json
import time

import numpy as np
import zmq
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (
    NetworkIntrinsics,
    postprocess_nanodet_detection,
)

ZMQ_ADDR = "tcp://*:5555"

WIDTH, HEIGHT = 416, 416

MODEL = (
    "/usr/share/imx500-models/"
    "imx500_network_nanodet_plus_416x416_pp.rpk"
)

# Reduce threshold so that send low score to tracker
POST_CONF = 0.50
SEND_THRESHOLD = 0.15

IOU = 0.65
MAX_DETECTIONS = 20


def clip_box(x1, y1, x2, y2):
    x1 = max(0.0, min(float(WIDTH - 1), float(x1)))
    y1 = max(0.0, min(float(HEIGHT - 1), float(y1)))
    x2 = max(0.0, min(float(WIDTH - 1), float(x2)))
    y2 = max(0.0, min(float(HEIGHT - 1), float(y2)))
    return x1, y1, x2, y2


def parse_detections(metadata):
    np_outputs = imx500.get_outputs(metadata, add_batch=True)

    # 这个不是“没检测到人”
    # 这是“这一帧没有新的IMX推理结果”
    if np_outputs is None:
        return [], False

    input_w, input_h = imx500.get_input_size()

    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0],
            conf=POST_CONF,
            iou_thres=IOU,
            max_out_dets=MAX_DETECTIONS,
        )[0]

        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(
            boxes,
            1,
            1,
            input_h,
            input_w,
            False,
            False,
        )

    else:
        boxes = np_outputs[0][0]
        scores = np_outputs[1][0]
        classes = np_outputs[2][0]

        if intrinsics.bbox_normalization:
            boxes = boxes / input_h

        if intrinsics.bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

    objs = []

    for box, score, cls in zip(boxes, scores, classes):
        if score < SEND_THRESHOLD:
            continue

        # 你实测后应按这个理解：
        # convert_inference_coords returns x, y, w, h
        x, y, w, h = imx500.convert_inference_coords(
            box,
            metadata,
            picam2,
        )

        x1 = float(x)
        y1 = float(y)
        x2 = float(x + w)
        y2 = float(y + h)

        x1, y1, x2, y2 = clip_box(x1, y1, x2, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        def accept_person_box(x1, y1, x2, y2, score, cls_id):
            if int(cls_id) != 0:
                return False

            w = x2 - x1
            h = y2 - y1
            area = w * h

            if w <= 0 or h <= 0:
                return False

            if h < 100:
                return False

            if area < 9000:
                return False

            # 人体框通常不能太扁，但你这个模型框比较方，所以这里只做宽松限制。
            aspect = w / h
            if aspect < 0.35 or aspect > 1.35:
                return False

            # 对底部小框做额外过滤：
            # 如果框从画面很低的位置才开始，而且高度又不够，基本不是完整人。
            if y1 > 200 and h < 140:
                return False

            return True

        if not accept_person_box(x1, y1, x2, y2, score, cls):
            continue

        objs.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "score": float(score),
            "cls": int(cls),
        })

    # det_valid=True 说明这一帧确实有IMX推理结果。
    # objs为空，才是真正的 no detections。
    return objs, True


def main():
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)

    # 队列小：tracker慢就丢旧帧，避免延迟堆积
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(ZMQ_ADDR)

    global imx500, intrinsics, picam2

    imx500 = IMX500(MODEL)

    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"

    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)


    from libcamera import Transform
    config = picam2.create_preview_configuration(
        main={
            "size": (WIDTH, HEIGHT),
            "format": "XRGB8888",
        },
        controls={
            "FrameRate": intrinsics.inference_rate,
        },
        transform=Transform(
            hflip=False,
            vflip=False,
        ),
        buffer_count=4,
    )

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    print("[IMX] started")
    print("[IMX] inference_rate:", intrinsics.inference_rate)
    print("[IMX] postprocess:", intrinsics.postprocess)
    print("[IMX] preserve_aspect_ratio:", intrinsics.preserve_aspect_ratio)

    time.sleep(1.0)

    frame_id = 0
    last_log_t = time.perf_counter()
    sent_count = 0

    try:
        while True:
            request = picam2.capture_request()

            try:
                frame = np.ascontiguousarray(
                    request.make_array("main")
                )
                metadata = request.get_metadata()

            finally:
                request.release()

            objs, det_valid = parse_detections(metadata)

            header = {
                "frame_id": frame_id,
                "shape": frame.shape,
                "dtype": "uint8",
                "objects": objs,
                "det_valid": det_valid,
                "send_ts_ns": time.perf_counter_ns(),
            }

            try:
                sock.send_multipart(
                    [
                        json.dumps(header).encode("utf-8"),
                        frame.tobytes(),
                    ],
                    flags=zmq.NOBLOCK,
                )
                sent_count += 1

            except zmq.Again:
                pass

            frame_id += 1

            now = time.perf_counter()
            if now - last_log_t >= 2.0:
                print(
                    f"[IMX] pub_fps={sent_count / (now - last_log_t):.1f} "
                    f"dets={len(objs)} "
                    f"det_valid={det_valid}"
                )

                if objs:
                    print("[IMX] first_det:", objs[0])

                sent_count = 0
                last_log_t = now

    except KeyboardInterrupt:
        print("\n[IMX] exiting...")

    finally:
        picam2.stop()
        picam2.close()
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
