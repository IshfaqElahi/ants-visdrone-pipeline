# ants-visdrone-pipeline
A computer vision pipeline utilizing YOLOv8 Nano and ByteTrack on the VisDrone dataset for real-time object detection, human counting, and multi-object video tracking in aerial surveillance.

# Post-Clone Setup Instructions

### Navigate to the Project Directory
```bash
cd ants-visdrone-pipeline
```
Set Up the Virtual Environment
It is highly recommended to run this project inside an isolated virtual environment to prevent library conflicts.
```bash
# Create the virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (Mac/Linux)
source .venv/bin/activate
```
 Install Dependencies
With your virtual environment activated, install the required packages:
```bash
pip install -r requirements.txt
```
### Download and Place External Data
Due to file size constraints, the 2.11 GB dataset is hosted externally on Google Drive. **🔗 [Google Drive Link - Download Here](https://drive.google.com/drive/folders/1MT7mrrRQhJosLVBaPmuaw1FVqgk75Km4?usp=sharing)**
1. Download the `dataset` folder from the provided Google Drive link.
2. Place the `dataset` folder directly into the root directory of this cloned project.

*(Note: The trained model weights, `best.pt`, are already included in this repository under the `runs/` folder, so you only need to download the dataset!)*

### ⚠️ Note on Large Video Files & Outputs

Due to GitHub's strict file size limits, the following large media files and output directories have been excluded from this repository:
* `Task04_Outputs/` (Directory containing the final tracked result videos)
* `test_video.mp4` (Original source video)
* `test_video_1.mp4` (Original source video)

**🔗 [Google Drive Link - Download Here](https://drive.google.com/drive/folders/1MT7mrrRQhJosLVBaPmuaw1FVqgk75Km4?usp=sharing)**

### ByteTrack Video Tracking
Opens a native UI file-picker to select a drone video, applies ByteTrack persistent temporal tracking, and saves the output video. You are not limited to the provided test videos! You can download *any* aerial footage of city traffic (e.g., from YouTube or Pexels), run this script, select your custom video from the file picker, and watch the pipeline track vehicles in real-time.

```bash
python task4_tracking.py
```

**Placement Instructions:**
1. Download the missing videos and folder from the Drive link above.
2. Place `test_video.mp4` and `test_video_1.mp4` directly into the **root directory** of this cloned project.
3. Place the `Task04_Outputs` folder directly into the **root directory** to view the final ByteTrack results (or simply run the tracking script to generate new outputs!).

### Run the Pipeline
The project uses dynamic relative paths, so the scripts will execute flawlessly once the data is in place.

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
ants-visdrone-pipeline/
├── dataset/VisDrone_Dataset/
│   ├── VisDrone2019-DET-train/images/       # Training images (.jpg)
│   ├── VisDrone2019-DET-train/labels/       # YOLO annotation files (.txt)
│   ├── VisDrone2019-DET-test-dev/images/    # Test images (.jpg)
│   └── visdrone.yaml                        # Dataset config for Ultralytics
├── runs/detect/Task02_Results/
│   └── yolov8n_run1-2/weights/
│       ├── best.pt                          # Best checkpoint (Tasks 3 & 4)
│       └── last.pt                          # Last checkpoint (Task 2 resume)
├── Task01_Visualizations/                   # Output: annotated training images
├── Task03_Outputs/                          # Output: inference images with counts
└── Task04_Outputs/                          # Output: tracked video files
```
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
```

Images absent from the label directory (background images) are handled gracefully — the script prints a note and renders the unannotated image without crashing.

OpenCV reads images in BGR. Colours are defined in BGR order. Matplotlib expects RGB, so a `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` conversion is applied before display.

---

# 3. Task 2 — Model Training

**Scripts:** `task2_train.py` | `task2_resume.py`

Task 2 fine-tunes a pre-trained YOLOv8 nano (`yolov8n`) model on the VisDrone dataset. Two scripts are provided: an initial training launcher and a checkpoint-resume utility.

## 3.1 Initial Training (`task2_train.py`)

### Training Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `model` | `yolov8n.pt` | Pre-trained COCO nano model used as the starting checkpoint |
| `data` | `visdrone.yaml` | Ultralytics dataset config (class names, train/val/test paths) |
| `epochs` | `10` | Number of complete passes through the training set |
| `imgsz` | `640` | Input resolution (pixels); standard YOLOv8 training size |
| `batch` | `16` | Number of images processed per gradient update step |
| `project` | `Task02_Results` | Root directory under `runs/detect/` for saving results |
| `name` | `yolov8n_run1` | Sub-directory name for this specific run |

Ultralytics automatically saves `weights/best.pt` (highest validation mAP) and `weights/last.pt` (most recent epoch) inside the named run directory.

## 3.2 Resuming Training (`task2_resume.py`)

The training was intentionally interrupted because it was showing a over night timing to finish training, `task2_resume.py` reloads `last.pt` and calls `model.train(resume=True)`. Ultralytics reads the original hyperparameters from the run's `args.yaml` file, restoring the optimizer state, learning-rate schedule, and epoch counter seamlessly.

```python
checkpoint_path = r'D:\ANTS\runs\detect\Task02_Results\yolov8n_run1-2\weights\last.pt'

model = YOLO(checkpoint_path)
model.train(resume=True)
```

## 3.3 Output Artefacts

- `weights/best.pt` — consumed by Task 3 (inference) and Task 4 (tracking)
- `weights/last.pt` — consumed by Task 2 resume
- `results.csv` — per-epoch loss and metric curves
- `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png` — validation diagnostics

---

# 4. Task 3 — Inference & Human Counting

**Script:** `task3_inference.py`

This script loads the trained model, runs single-image inference on a randomly selected test image, draws labelled bounding boxes, overlays a total human count, saves the output image, and displays it in a resizable OpenCV window.

## 4.1 Configuration

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `model_path` | str (path) | Path to `best.pt` produced by Task 2 |
| `test_images_dir` | str (path) | VisDrone 2019 DET test-dev images directory |
| `output_dir` | str (path) | Directory for saving output images; auto-created |

## 4.2 Detection & Counting Logic

For each detected bounding box the script checks `cls_id`:

```python
if cls_id in [0, 1]:        # Pedestrian or People
    human_count += 1
    color = (0, 0, 255)     # Red (BGR)
    label = f'Human {conf:.2f}'

elif cls_id == 3:           # Car
    color = (0, 255, 0)     # Green (BGR)
    label = f'Car {conf:.2f}'

else:
    continue                # All other classes are skipped
```

## 4.3 On-screen Counter Overlay

A black filled rectangle is drawn first as a background, then white text is rendered on top, ensuring readability against any scene background:

```python
count_text = f'Total Humans Counted: {human_count}'

(text_width, text_height), _ = cv2.getTextSize(
    count_text,
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    2
)

cv2.rectangle(
    img,
    (20, 20),
    (40 + text_width, 40 + text_height + 10),
    (0, 0, 0),
    -1
)

cv2.putText(
    img,
    count_text,
    (30, 30 + text_height),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)
```

## 4.4 Output

- Saved image: `counted_<original_filename>.jpg` in `Task03_Outputs/`
- Console log: inference target filename and detected human count
- Interactive window: `1280×720` resizable OpenCV display, closed by any key press

---

# 5. Task 4 — ByteTrack Video Tracking

**Script:** `task4_tracking.py`

Task 4 extends the detection pipeline to video. It launches a native OS file-picker dialog, loads the trained model, processes each frame with ByteTrack persistent tracking, renders annotated frames, and saves the output as a new video file.

## 5.1 Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `model_path` | str (path) | Path to `best.pt` produced by Task 2 |
| `output_dir` | str (path) | Directory for tracked video output; auto-created |
| `tracker` | `bytetrack.yaml` | Ultralytics built-in ByteTrack configuration file |
| `persist` | `True` | Maintains track IDs across frames for continuity |
| `verbose` | `False` | Suppresses per-frame console output for performance |

## 5.2 GUI File Selection

Tkinter is used solely to display the OS-native file-picker dialog. The root window is immediately hidden (`root.withdraw()`) so no blank Tkinter window appears. If the user cancels the dialog, the program exits gracefully.

```python
root = tk.Tk()
root.withdraw()    # Hides the blank Tkinter root window

video_path = filedialog.askopenfilename(
    title='Select a Drone Video to Track',
    filetypes=[
        ('Video Files', '*.mp4 *.avi *.mov *.mkv'),
        ('All Files', '*.*')
    ]
)

if not video_path:  # User pressed Cancel
    return
```

## 5.3 Frame-by-Frame Processing

```python
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker='bytetrack.yaml',
        verbose=False
    )

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    cv2.imshow('Bonus Task - ByteTrack', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## 5.4 Output Naming Convention

The output file is automatically named by appending `_tracked` to the original filename stem, preserving the container format:

```python
# Input:  drone_footage.mp4
# Output: drone_footage_tracked.mp4

name_only, ext = os.path.splitext(os.path.basename(video_path))

output_path = os.path.join(
    output_dir,
    f'{name_only}_tracked{ext}'
)
```

## 5.5 Video Writer Properties

The output video writer mirrors the source properties exactly:

- FPS: extracted from `cv2.CAP_PROP_FPS`
- Resolution: extracted from `CAP_PROP_FRAME_WIDTH` and `CAP_PROP_FRAME_HEIGHT`
- Codec: `mp4v` (MPEG-4 Part 2); compatible with `.mp4` containers

---

# 6. Dependencies & Environment Setup

## 6.1 Required Packages

| Package | Version | Description |
| :--- | :--- | :--- |
| `ultralytics` | `>=8.0` | YOLOv8 training, inference, and ByteTrack tracking |
| `opencv-python` | `>=4.5` | Image/video I/O, drawing, and window display |
| `matplotlib` | `>=3.5` | Interactive image display (Task 1 only) |
| `tkinter` | stdlib | GUI file dialog (Task 4); included with Python |

## 6.2 Installation

```bash
pip install ultralytics opencv-python matplotlib
```

## 6.3 Hardware Recommendations

- GPU: NVIDIA GPU with CUDA 11.8+ strongly recommended for training (Task 2)
- RAM: Minimum 8 GB system RAM; 16 GB recommended
- Disk: ~5 GB for VisDrone dataset; ~500 MB for model weights and outputs

CPU-only inference is supported for Tasks 3 and 4, though slower.

---

# 7. Common Issues & Resolutions

| Issue | Resolution |
| :--- | :--- |
| No images found in folder | Check that `img_dir` and `test_images_dir` paths are correct and contain `.jpg` files |
| `FileNotFoundError` on `.pt` file | Ensure training completed and `best.pt` / `last.pt` exist at the specified `model_path` |
| CUDA out of memory | Reduce batch size in `task2_train.py` (e.g., from `16` to `8` or `4`) |
| OpenCV window not responding | Press any key to close; avoid closing via window X button mid-run |
| Tkinter dialog not appearing | Ensure a display server is active; remote SSH sessions require X11 forwarding |
| Video output is corrupted | Verify that the input video codec is supported; try converting to H.264 MP4 first |
| Resume fails with `KeyError` | Delete the existing run folder and restart training from scratch |

---

# 8. Quick Reference

| Script | Command | Description |
| :--- | :--- | :--- |
| `task1_visualization.py` | `python task1_visualization.py` | Randomly picks & annotates a training image |
| `task2_train.py` | `python task2_train.py` | Starts fresh YOLOv8n training run |
| `task2_resume.py` | `python task2_resume.py` | Resumes interrupted training from `last.pt` |
| `task3_inference.py` | `python task3_inference.py` | Runs inference on a random test image & counts humans |
| `task4_tracking.py` | `python task4_tracking.py` | Opens file picker, tracks objects in selected video |
