# CodeAlpha Object Detection and Tracking

This project implements a real-time object detection and tracking system using OpenCV and a pre-trained YOLOv3 model. It was developed as part of the CodeAlpha Artificial Intelligence Internship.

## 🌟 Features
- **Real-Time Detection:** Captures live video feed from your webcam and performs object detection frame-by-frame.
- **YOLOv3 Model:** Utilizes the robust YOLOv3 (You Only Look Once) deep learning model for high-accuracy bounding box predictions.
- **Object Tracking:** Implements a custom tracking algorithm based on the Euclidean distance between bounding box centroids to assign and maintain persistent IDs for objects as they move.
- **Non-Maximum Suppression (NMS):** Filters out overlapping bounding boxes to ensure clean and distinct object detections.

## 🛠️ Technologies Used
- Python 3.x
- `opencv-python` (cv2) for video capture, DNN module, and image processing.
- `numpy` for matrix and distance calculations.
- Pre-trained YOLOv3 Weights and Config.

## 🚀 How to Run

1. **Prerequisites:**
   Ensure you have the YOLOv3 weights file (`yolov3.weights`). *Note: Due to its size, you may need to download it separately if it is not included in the repository.*

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Tracker:**
   ```bash
   python main.py
   ```

4. **Usage:**
   - The application will open a window showing your webcam feed.
   - Detected objects will be highlighted with bounding boxes, class labels, and unique tracking IDs.
   - Press the `q` key while the window is focused to quit the application.
