import os
from ultralytics import YOLO

def resume_training():
    # 1. Define project-relative folders
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    checkpoint_path = os.path.join(
        BASE_DIR, 
        'runs', 
        'detect', 
        'Task02_Results', 
        'yolov8n_run1-2', 
        'weights', 
        'last.pt'
    )

    # --- SAFETY CHECK ---
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find the training checkpoint at '{checkpoint_path}'.")
        print("Cannot resume training because 'last.pt' is missing. You may need to start a fresh training run using 'task2_train.py'.")
        return
    
    # Load the checkpoint
    model = YOLO(checkpoint_path)

    # Resume the training process
    model.train(resume=True)

if __name__ == '__main__':
    resume_training()