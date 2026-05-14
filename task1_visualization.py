import cv2
import matplotlib.pyplot as plt
import os
import random

def visualize_random_yolo_sample():
    # 1. Define project-relative folders
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    img_dir = os.path.join(
        BASE_DIR,
        'dataset',
        'VisDrone_Dataset',
        'VisDrone2019-DET-train',
        'images'
    )

    label_dir = os.path.join(
        BASE_DIR,
        'dataset',
        'VisDrone_Dataset',
        'VisDrone2019-DET-train',
        'labels'
    )
    
    # Output folder
    output_dir = os.path.join(BASE_DIR, 'Task01_Visualizations')
    os.makedirs(output_dir, exist_ok=True) # This creates the folder if it doesn't exist
    
    # 2. Check if the dataset exists before proceeding to prevent crashes
    if not os.path.exists(img_dir):
        print(f"Error: Could not find the dataset folder at '{img_dir}'.")
        print("Please ensure you have downloaded the VisDrone dataset and placed it in the 'dataset' folder.")
        return

    # Get a list of all .jpg files in the image folder
    all_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    if not all_images:
        print("Error: No images found in the folder!")
        return

    # 3. Pick a random image!
    random_image = random.choice(all_images)
    print(f"Automatically selected: {random_image}")
    
    # 4. Construct the exact paths for the image and its matching .txt label
    image_path = os.path.join(img_dir, random_image)
    label_path = os.path.join(label_dir, random_image.replace('.jpg', '.txt'))

    # Load image (Defaults to BGR format in OpenCV)
    img = cv2.imread(image_path)
    h_img, w_img, _ = img.shape
    
    # Target classes with BGR colors for correct OpenCV saving
    target_classes = {
        0: ('Human (Pedestrian)', (0, 0, 255)), # Red in BGR
        1: ('Human (People)', (0, 0, 255)),     # Red in BGR
        3: ('Car', (0, 255, 0))                 # Green in BGR
    }
    
    # Check if the label file exists before trying to open it
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            data = line.strip().split()
            if len(data) < 5: continue
                
            class_id = int(data[0])
            if class_id in target_classes:
                x_center, y_center = float(data[1]), float(data[2])
                w, h = float(data[3]), float(data[4])
                
                label, color = target_classes[class_id]
                x1 = int((x_center - w / 2) * w_img)
                y1 = int((y_center - h / 2) * h_img)
                box_w = int(w * w_img)
                box_h = int(h * h_img)
                
                cv2.rectangle(img, (x1, y1), (x1 + box_w, y1 + box_h), color, 2)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        print(f"Note: No labels found for {random_image} (It might be an empty background image)")

    # Save the image to your output folder
    save_path = os.path.join(output_dir, f"detected_{random_image}")
    cv2.imwrite(save_path, img)
    print(f"Successfully saved visualized image to: {save_path}")

    # Show the final image using Matplotlib
    # Convert BGR to RGB so Matplotlib displays the colors correctly
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"Random Sample: {random_image} | Humans (Red) & Cars (Green)")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_random_yolo_sample()