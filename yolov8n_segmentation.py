from ultralytics import YOLO

class YOLOv8nSegmentation:
    def __init__(self, model_path='yolov8n-seg.pt'):
        self.model = YOLO(model_path)
    
    def segment(self, image):
        # Validate the model
        results = self.model(image)
        return results
