python main.py \
  --detector yolo \
  --source rtsp://127.0.0.1:8554/imx1 \
  --output rtsp://127.0.0.1:8554/imx1det \
  --width 320 --height 320 \
  --det_model models/yolo26n.mnn \
  --reid_model_name osnet_x1_0 \
  --reid_weights weights/osnet_x1_0_market.pth \
  --camera_id imx1 \
  --emit_tracks --transport zmq \
  --track_zmq_bind tcp://192.168.3.39:6001

