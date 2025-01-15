from ultralytics import YOLO

# Load a model
model = YOLO("/home/eeproj7/idan_carmel/yolo11l.pt")  

# Train the model
results = model.train(data="/home/eeproj7/idan_carmel/Drone_merge/data.yaml", epochs=1500, imgsz=640, save=True)
