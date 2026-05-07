from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("yolo26n.pt")

# Export the model to MNN format
model.export(format="mnn", imgsz=1440)

# Load the exported MNN model
mnn_model = YOLO("yolo26n.mnn")

# Run inference
results = mnn_model("data/00000000.png")