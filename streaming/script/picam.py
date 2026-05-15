"""rpicam does not support straightforward single-process V4L2 handling.
Instead, it relies on a FIFO-based pipeline to separate frame reading and writing.

In addition, using FFmpeg directly for decoding is unreliable here—input resolution control is fragile and
may trigger pixel format parsing errors without clear causes.

Therefore, we adopt the official Picamera2 toolkit rather than manually managing FFmpeg.
This is why the pipeline cannot be simplified into a shell script
like those used for macOS cameras or standard USB webcams."""
import time

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

#WIDTH, HEIGHT = 640, 480 # calibration
WIDTH, HEIGHT = 320, 320 # detection sample
FPS = 30
BITRATE = 3_000_000
RTSP_URL = "rtsp://127.0.0.1:8554/cam0"

picam2 = Picamera2()

# Configure video stream
config = picam2.create_video_configuration(
    main={"size": (WIDTH, HEIGHT), "format": "YUV420"},
    controls={"FrameRate": FPS}
)
picam2.configure(config)

# ---- Use simple Libav H.264 encoder (compatible with all backends) ----
encoder = H264Encoder(bitrate=BITRATE)

# ffmpeg output to RTSP
ffout = FfmpegOutput(f"-f rtsp -rtsp_transport tcp {RTSP_URL}", audio=False)

picam2.start_recording(encoder, ffout)
print(f"Streaming {WIDTH}x{HEIGHT}@{FPS} to {RTSP_URL} (Ctrl+C to stop)")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    picam2.stop_recording()
