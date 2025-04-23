from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  # use yolov8s.pt for better accuracy
    model.train(data="candy.yaml", epochs=50, imgsz=640, batch=8)
