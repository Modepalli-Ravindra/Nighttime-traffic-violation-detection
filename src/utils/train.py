from ultralytics import YOLO
from roboflow import Roboflow
import os

def train_model():
    # --- 1. Download Dataset from Roboflow ---
    rf = Roboflow(api_key="2e2L1iECldYWRfethjxg")
    project = rf.workspace("adsfghm").project("my-first-project-nlhlo")
    version = project.version(1)
    dataset = version.download("yolov8")

    print(f"Dataset downloaded to: {dataset.location}")

    # --- 2. Train YOLOv8 Model ---
    
    # Load model
    model = YOLO("yolov8n.pt")  # load a pretrained model (nano for speed)
    
    print("\n🔥 Starting Training... (This may take a while)")
    # Train the model
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=30,           # Quick training for demo (increase to 100 for better results)
        imgsz=640,
        plots=True,
        batch=4,             # Small batch size for laptop
        name="traffic_night_model"
    )
    
    print("\n🎉 Training Complete!")
    print(f"Best model saved at: runs/detect/traffic_night_model/weights/best.pt")

if __name__ == "__main__":
    train_model()
