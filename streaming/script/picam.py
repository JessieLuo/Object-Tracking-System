import argparse
import signal
import sys
import time
from threading import Event

from libcamera import Transform
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

# ----------------------------------------
# GLOBAL
# ----------------------------------------

stop_event = Event()

# ----------------------------------------
# CLI
# ----------------------------------------

parser = argparse.ArgumentParser()

# camera list

parser.add_argument(
    "--camera",
    type=int,
    nargs="+",
    default=[0],
    help="Camera ids, e.g. --camera 0 1"
)

# resolution

parser.add_argument(
    "--width",
    type=int,
    default=320,
    help="Frame width"
)

parser.add_argument(
    "--height",
    type=int,
    default=320,
    help="Frame height"
)

# fps

parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Frame rate"
)

# bitrate

parser.add_argument(
    "--bitrate",
    type=int,
    default=3_000_000,
    help="H264 bitrate"
)

# rtsp server

parser.add_argument(
    "--host",
    type=str,
    default="127.0.0.1",
    help="RTSP server host"
)

parser.add_argument(
    "--port",
    type=int,
    default=8554,
    help="RTSP server port"
)

# flip

parser.add_argument(
    "--hflip",
    action="store_true"
)

parser.add_argument(
    "--vflip",
    action="store_true"
)

args = parser.parse_args()

# ----------------------------------------
# STREAM HOLDER
# ----------------------------------------

streams = []

# ----------------------------------------
# CREATE STREAMS
# ----------------------------------------

for cam_id in args.camera:

    print(f"Opening camera {cam_id}")

    picam2 = Picamera2(camera_num=cam_id)

    config = picam2.create_video_configuration(
        main={
            "size": (
                args.width,
                args.height
            ),
            "format": "YUV420",
        },
        controls={
            "FrameRate": args.fps,
        },
        transform=Transform(
            hflip=args.hflip,
            vflip=args.vflip,
        )
    )

    picam2.configure(config)

    encoder = H264Encoder(
        bitrate=args.bitrate
    )

    rtsp_url = (
        f"rtsp://"
        f"{args.host}:"
        f"{args.port}/"
        f"imx{cam_id}"
    )

    ffout = FfmpegOutput(
        f"-f rtsp "
        f"-rtsp_transport tcp "
        f"{rtsp_url}",
        audio=False
    )

    picam2.start_recording(
        encoder,
        ffout
    )

    print(
        f"[CAM {cam_id}] "
        f"{args.width}x{args.height} "
        f"{args.fps}fps "
        f"-> {rtsp_url}"
    )

    streams.append(
        {
            "id": cam_id,
            "picam2": picam2,
        }
    )

# ----------------------------------------
# CLEAN EXIT
# ----------------------------------------

def shutdown(sig=None, frame=None):

    print("\nStopping streams...")

    for s in streams:

        try:
            s["picam2"].stop_recording()

        except Exception as e:
            print(
                f"Camera {s['id']} stop failed: {e}"
            )

    stop_event.set()

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ----------------------------------------
# LOOP
# ----------------------------------------

try:

    while not stop_event.is_set():
        time.sleep(1)

except KeyboardInterrupt:
    shutdown()

sys.exit(0)