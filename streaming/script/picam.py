"""rpicam does not support straightforward single-process V4L2 handling.
Instead, it relies on a FIFO-based pipeline to separate frame reading and writing.

In addition, using FFmpeg directly for decoding is unreliable here—input resolution control is fragile and
may trigger pixel format parsing errors without clear causes.

Therefore, we adopt the official Picamera2 toolkit rather than manually managing FFmpeg.
This is why the pipeline cannot be simplified into a shell script
like those used for macOS cameras or standard USB webcams."""
import time
import argparse

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

from libcamera import Transform

# WIDTH, HEIGHT = 640, 480
WIDTH, HEIGHT = 320, 320

FPS = 30
BITRATE = 3_000_000
RTSP_URL = "rtsp://127.0.0.1:8554/cam0"

# --------------------------------------------------
# argparse
# default = False
# user must explicitly enable flips
# --------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "--hflip",
    action="store_true",
    help="Enable horizontal flip"
)

parser.add_argument(
    "--vflip",
    action="store_true",
    help="Enable vertical flip"
)

args = parser.parse_args()

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={
        "size": (WIDTH, HEIGHT),
        "format": "YUV420",
    },
    controls={
        "FrameRate": FPS,
    },
    transform=Transform(
        hflip=args.hflip,
        vflip=args.vflip,
    )
)

picam2.configure(config)

encoder = H264Encoder(bitrate=BITRATE)

ffout = FfmpegOutput(
    f"-f rtsp -rtsp_transport tcp {RTSP_URL}",
    audio=False
)

picam2.start_recording(encoder, ffout)

print(
    f"Streaming {WIDTH}x{HEIGHT}@{FPS} "
    f"hflip={args.hflip} "
    f"vflip={args.vflip}"
)

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    picam2.stop_recording()
