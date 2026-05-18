from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("../models/yolo26n.pt")

# Export the model to MNN format
model.export(format="mnn", imgsz=320, project="../models")

# Load the exported MNN model
mnn_model = YOLO("../models/yolo26n.mnn", task="detect")

# Run inference
results = mnn_model("data/00000000.png")