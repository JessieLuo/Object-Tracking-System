from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("yolo26n.pt")

# Export the model to ONNX format
model.export(format="onnx", imgsz=320)

# Load the exported ONNX model
onnx_model = YOLO("yolo26n.onnx")

# Run inference
results = onnx_model("../track/data/bus.jpg")