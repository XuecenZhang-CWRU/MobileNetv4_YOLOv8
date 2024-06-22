
from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    # model = YOLO('yolov8m.pt')  # load an official model
    model = YOLO('runs/detect/train/weights/best.pt')  # load a custom model


    # Validate the model
    metrics = model.val(split='val')  # no arguments needed, dataset and settings remembered

