from ultralytics import YOLO

if __name__ == "__main__":
    print(open("candy.yaml").read())
    model = YOLO("yolov8s.pt") 
    model.train(data="candy.yaml", epochs=100, imgsz=640, batch=8, device="cpu")
