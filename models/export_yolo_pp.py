from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("yolo26n.pt")

# Export the model to PaddlePaddle format
model.export(format="paddle", imgsz=320)

# Load the exported PaddlePaddle model
paddle_model = YOLO("./yolo26n_paddle_model") # slowest among the 4 formats

# Run inference
results = paddle_model("../track/data/bus.jpg")