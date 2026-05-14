import cv2
import os
import random
from ultralytics import YOLO

def run_detection_and_count():
    # 1. Define project-relative folders
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

    test_images_dir = os.path.join(
        BASE_DIR, 
        'dataset', 
        'VisDrone_Dataset', 
        'VisDrone2019-DET-test-dev', 
        'images'
    )
    
    # 2. Setup the saving mechanism
    output_dir = os.path.join(BASE_DIR, 'Task03_Outputs')
    os.makedirs(output_dir, exist_ok=True)

    # --- SAFETY CHECKS ---
    if not os.path.exists(test_images_dir):
        print(f"Error: Could not find the test images folder at '{test_images_dir}'.")
        print("Please ensure you have downloaded the VisDrone dataset from the Google Drive link and placed it in the 'dataset' folder.")
        return

    if not os.path.exists(model_path):
        print(f"Error: Could not find the trained model at '{model_path}'.")
        print("Please ensure you have downloaded the 'best.pt' file from the Google Drive link and placed it in the correct folder.")
        return

    # 3. Pick a random test image to evaluate
    all_images = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')]
    if not all_images:
        print("Error: No images found in test folder!")
        return
    
    random_image_name = random.choice(all_images)
    image_path = os.path.join(test_images_dir, random_image_name)
    print(f"Running inference on: {random_image_name}")

    # 4. Load your fully trained model
    model = YOLO(model_path)

    # 5. Run Inference
    results = model(image_path)[0]

    # 6. Load the image with OpenCV for drawing
    img = cv2.imread(image_path)
    human_count = 0

    # 7. Process Detections
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Logic for humans (Class 0: pedestrian, Class 1: people)
        if cls_id in [0, 1]: 
            human_count += 1
            color = (0, 0, 255)  # Red in BGR
            label = f"Human {conf:.2f}"
        # Logic for cars (Class 3: car)
        elif cls_id == 3:      
            color = (0, 255, 0)  # Green in BGR
            label = f"Car {conf:.2f}"
        else:
            continue  # Skip anything else

        # Draw the Bounding Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 8. Display the Total Human Count
    count_text = f"Total Humans Counted: {human_count}"
    
    # Draw a black background rectangle for the text to make it readable
    (text_width, text_height), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(img, (20, 20), (40 + text_width, 40 + text_height + 10), (0, 0, 0), -1)
    cv2.putText(img, count_text, (30, 30 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 9. Save the Output
    save_path = os.path.join(output_dir, f"counted_{random_image_name}")
    cv2.imwrite(save_path, img)
    print(f"Success: Found {human_count} humans. Image saved to: {save_path}")

    # 10. Show the Result on Screen
    # OpenCV window is resizable so it fits your screen
    cv2.namedWindow("Drone Surveillance - Task 03", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Surveillance - Task 03", 1280, 720)
    cv2.imshow("Drone Surveillance - Task 03", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_detection_and_count()