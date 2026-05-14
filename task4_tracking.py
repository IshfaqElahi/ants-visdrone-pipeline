import cv2
import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

def run_tracking_with_ui():
    # 1. Setup the Windows Pop-up Dialog
    root = tk.Tk()
    root.withdraw()  # This hides the main blank Tkinter window
    
    print("Opening file explorer... Please select a video.")
    
    # Open the file dialog box
    video_path = filedialog.askopenfilename(
        title="Select a Drone Video to Track",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )
    
    # If the user clicks "Cancel" on the pop-up, stop the program gracefully
    if not video_path:
        print("No video selected. Exiting program...")
        return

    print(f"Selected video: {video_path}")

    # 2. Define project-relative folders
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(
        BASE_DIR, 
        'runs', 
        'detect', 
        'Task02_Results', 
        'yolov8n_run1-2', 
        'weights', 
        'best.pt'
    )
    
    output_dir = os.path.join(BASE_DIR, 'Task04_Outputs')
    os.makedirs(output_dir, exist_ok=True)

    # --- SAFETY CHECK ---
    if not os.path.exists(model_path):
        print(f"Error: Could not find the trained model at '{model_path}'.")
        print("Please ensure you have downloaded the 'best.pt' file from the Google Drive link and placed it in the correct folder.")
        return
    
    # Extract the original name and create a new safe save name
    base_name = os.path.basename(video_path)
    name_only, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name_only}_tracked{ext}")

    # 3. Load your fully trained model
    model = YOLO(model_path)

    # 4. Open the Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the selected video.")
        return

    # Get video properties for saving the output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Starting ByteTrack... Saving output to: {output_path}")
    print("Press 'q' on your keyboard to stop early.")

    # 5. Process the Video Frame by Frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break # Video is finished

        # Run ByteTrack Tracking!
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # YOLOv8 has a built-in plotter
        annotated_frame = results[0].plot()

        # Save the frame to our new video
        out.write(annotated_frame)

        # Show the live feed on screen
        cv2.namedWindow("Bonus Task - ByteTrack", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Bonus Task - ByteTrack", 1280, 720)
        cv2.imshow("Bonus Task - ByteTrack", annotated_frame)

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Tracking complete! Video saved successfully.")

if __name__ == '__main__':
    run_tracking_with_ui()