python main.py \
  --detector imx_zmq \
  --zmq_addr tcp://127.0.0.1:5555 \
  --output rtsp://127.0.0.1:8554/imxdet \
  --width 320 --height 320 \
  --fps 26 \
  --reid_model_name osnet_x0_25 \
  --reid_weights weights/osnet_x0_25_msmt17.pt