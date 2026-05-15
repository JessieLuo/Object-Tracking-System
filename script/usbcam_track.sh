python main.py \
  --detector yolo \
  --source rtsp://127.0.0.1:8554/usbcam \
  --output rtsp://127.0.0.1:8554/usbdet \
  --width 320 --height 320 \
  --det_model models/yolo26n.mnn \
  --reid_model_name osnet_x1_0 \
  --reid_weights weights/osnet_x1_0_market.pth