#!/usr/bin/env bash
#set -euo pipefail

# !Notice: Open mediamtx first #
# Using Macbook web-camera as RTSP entry test
# critical param for buffer issue: -b -g
ffmpeg -loglevel warning \
  -f avfoundation -use_wallclock_as_timestamps 1 \
  -pixel_format nv12 -framerate 30 -video_size 640x480 -i "0" \
  -thread_queue_size 512 -r 30 \
  -vf format=nv12 \
  -c:v libx264 -preset veryfast -tune zerolatency \
  -profile:v baseline -level 3.0 \
  -b:v 600k -maxrate 600k -bufsize 300k \
  -g 30 -sc_threshold 0 \
  -x264-params "bframes=0:ref=1:rc-lookahead=0:repeat-headers=1:keyint=30:min-keyint=30:scenecut=0" \
  -an -f rtsp -rtsp_transport tcp -rtsp_flags listen \
  rtsp://0.0.0.0:8554/maccam
