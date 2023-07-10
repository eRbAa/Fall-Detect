from ultralytics import YOLO
if __name__ == '__main__':
 model = YOLO('Z:\homework\DXQ\yolo_test1\yolo_test1\yolov8n.pt')
 model.train(data='Z:\homework\DXQ\yolo_test1\yolo_test1\data/fall.yaml', batch=8,epochs=100,imgsz=640)