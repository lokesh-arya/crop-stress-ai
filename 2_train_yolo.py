from ultralytics import YOLO

# Load a pre-trained model (YOLOv8 Nano)
model = YOLO('yolov8n.pt')

# Define the training arguments
results = model.train(
    data='leaf_detection.yaml',  # Your dataset config file
    epochs=10,                   # Number of epochs
    imgsz=640,                   # Input image size
    batch=8,                     # Batch size (adjust based on GPU)
    name='stage1_yolov8n'        # Name for the output run
)
