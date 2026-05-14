# ants-visdrone-pipeline
An computer vision pipeline utilizing YOLOv8 Nano and ByteTrack on the VisDrone dataset for real-time object detection, human counting, and multi-object video tracking in aerial surveillance.
# ANTS PROJECT

# 1.1 Pipeline Architecture
The four tasks form a sequential workflow:

| Task | Description |
| :--- | :--- |
| **Task 1** | **Dataset Visualization** — Renders ground-truth YOLO annotations on random training images to verify label integrity before training. |
| **Task 2** | **Model Training** — Fine-tunes a pre-trained YOLOv8n model on the VisDrone dataset with checkpointing and resume support. |
| **Task 3** | **Inference & Counting** — Loads the trained model, runs inference on test images, draws bounding boxes, and reports human counts. |
| **Task 4** | **ByteTrack Video Tracking** — Applies persistent multi-object tracking to drone video footage with a GUI file-picker interface. |

### 1.2 Target Object Classes

| Class ID | Target | Description |
| :--- | :--- | :--- |
| `0` | Pedestrian | Single upright human figure; drawn in red |
| `1` | People | Group/clustered humans; drawn in red |
| `3` | Car | Motor vehicle; drawn in green |

### 1.3 Technology Stack
* **Python 3.8+**
* **Ultralytics YOLOv8** (`ultralytics` library)
* **OpenCV** (`cv2`) — image/video I/O and annotation drawing
* **Matplotlib** — interactive display for Task 1
* **Tkinter** — native OS file-picker dialog for Task 4
* **ByteTrack** — real-time multi-object tracking algorithm (built into Ultralytics)

### 1.4 Directory Structure
```text
D:\ANTS\
├── dataset\VisDrone_Dataset\
│   ├── VisDrone2019-DET-train\images\      # Training images (.jpg)
│   ├── VisDrone2019-DET-train\labels\      # YOLO annotation files (.txt)
│   ├── VisDrone2019-DET-test-dev\images\   # Test images (.jpg)
│   └── visdrone.yaml                        # Dataset config for Ultralytics
├── runs\detect\Task02_Results\
│   └── yolov8n_run1-2\weights\
│       ├── best.pt                          # Best checkpoint (used by Tasks 3 & 4)
│       └── last.pt                          # Last checkpoint (used by Task 2 resume)
├── Task01_Visualizations\                  # Output: annotated training images
├── Task03_Outputs\                         # Output: inference images with counts
└── Task04_Outputs\                         # Output: tracked video files

## 2. Task 1 — Dataset Visualization
**Script:** `task1_visualization.py`

This script randomly selects a training image from the VisDrone dataset, parses its corresponding YOLO annotation file, and renders colored bounding boxes for humans (red) and cars (green). The annotated image is saved to disk and displayed interactively via Matplotlib.

### 2.1 Configuration
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `img_dir` | str (path) | Absolute path to the training images directory |
| `label_dir` | str (path) | Absolute path to the YOLO label (`.txt`) files directory |
| `output_dir` | str (path) | Destination folder for saved annotated images; auto-created if absent |

### 2.2 Execution Flow
1. Scans `img_dir` for all `.jpg` files and selects one at random.
2. Resolves the matching `.txt` label file by replacing the extension.
3. Reads each annotation line: `class_id x_center y_center width height` (normalized).
4. Converts normalized YOLO coordinates to absolute pixel coordinates using image dimensions.
5. Draws `cv2.rectangle` and `cv2.putText` overlays with class-specific colours (BGR).
6. Saves the annotated image as `detected_<filename>.jpg` in `output_dir`.
7. Converts BGR to RGB and renders the result using Matplotlib for interactive viewing.

### 2.3 Key Implementation Notes
YOLO coordinate conversion formula:
```python
x1 = int((x_center - w / 2) * image_width)
y1 = int((y_center - h / 2) * image_height)
box_w = int(w * image_width)
box_h = int(h * image_height)
