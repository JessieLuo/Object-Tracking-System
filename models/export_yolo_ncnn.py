from ultralytics import YOLO

model = YOLO("yolo26n.pt", task="detect")
model.export(format="ncnn", imgsz=(640,480))

# Load the exported NCNN model
ncnn_model = YOLO("./yolo26n_ncnn_model")

# Run inference
results = ncnn_model("data/bus.jpg")