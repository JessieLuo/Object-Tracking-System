from docs.conf import project
from ultralytics import YOLO

model = YOLO("../models/yolo26n.pt", task="detect")
model.export(format="ncnn", imgsz=(640,480), project="../models")

# Load the exported NCNN model
ncnn_model = YOLO("../models/yolo26n_ncnn_model")

# Run inference
results = ncnn_model("data/bus.jpg")