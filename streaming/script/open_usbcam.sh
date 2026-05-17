ffmpeg \
  -f v4l2 \
  -input_format yuyv422 \
  -video_size 320x320 \
  -framerate 15 \
  -i /dev/video8 \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -f rtsp \
  rtsp://127.0.0.1:8554/usbcam
