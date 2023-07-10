from ultralytics import YOLO
 
# 加载模型
model = YOLO("Z:\homework\DXQ\yolo_test1\yolo_test1/fall_detect.pt")
 
# 转换模型
model.export(format="onnx")