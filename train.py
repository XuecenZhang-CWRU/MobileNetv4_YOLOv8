from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 加载模型
    model = YOLO(model="ultralytics/cfg/models/v8/yolov8l.yaml")  # 从头开始构建新模型

    # Use the model
    results = model.train(data="VOC.yaml", epochs=300, device='0', batch=8, seed=42,)  # 训练模
