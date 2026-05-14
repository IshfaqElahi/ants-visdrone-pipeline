import os
from ultralytics import YOLO

def train_visdrone():
    # 1. Define project-relative folders
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(BASE_DIR, 'dataset', 'VisDrone_Dataset', 'visdrone.yaml')

    # --- SAFETY CHECK ---
    if not os.path.exists(yaml_path):
        print(f"Error: Could not find the dataset configuration file at '{yaml_path}'.")
        print("Please ensure you have downloaded the VisDrone dataset from Google Drive and placed it in the 'dataset' folder.")
        return

    # 2. Load the pre-trained YOLOv8 'nano' model 
    model = YOLO('yolov8n.pt') 

    # 3. Train the model on your VisDrone dataset
    print("Starting YOLOv8 Model Training...")
    results = model.train(
        # Point this strictly to your yaml file
        data=yaml_path,
        
        # Training Parameters
        epochs=10,                 # 10 epochs for a quick assessment run
        imgsz=640,                 # Standard image size for YOLO
        batch=16,                  # Number of images processed at once
        project='Task02_Results',  # Main folder to save results
        name='yolov8n_run1',       # Sub-folder for this specific training run
    )
    print("Training Complete!")

if __name__ == '__main__':
    train_visdrone()