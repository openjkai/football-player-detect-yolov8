# Football Player Detection using YOLOv8

![Football Player Detection](https://img.shields.io/badge/Computer%20Vision-Object%20Detection-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-brightgreen)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)

## Overview

This project demonstrates how to detect football players in video footage using YOLOv8, a state-of-the-art real-time object detection model. The implementation leverages the Ultralytics YOLOv8 framework to identify and track players on the field.

## 🎬 Demo

[![Football Player Detection Demo](https://img.youtube.com/vi/AXp5uXWnYQA/0.jpg)](https://youtu.be/AXp5uXWnYQA)
*Click on the image above to watch the demo video*

## 🚀 Features

- **Football Player Detection**: Accurately identifies football players in video streams
- **Real-time Processing**: Optimized for efficient video processing
- **Pre-trained Model**: Utilizes YOLOv8's pre-trained weights
- **Easy Setup**: Simple implementation with minimal dependencies
- **Visualization**: Creates annotated output video with bounding boxes around detected players

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/openjkai/football-player-detect-yolov8.git
cd football-player-detect-yolov8

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install required packages
pip install ultralytics opencv-python
```

## 📊 Usage

The project is organized as a Jupyter notebook which can be run in environments like Google Colab, Jupyter Lab, or VSCode.

1. Open the notebook `Football_Player_Detection_YOLOv8.ipynb`
2. Install the required dependencies (if not already installed)
3. Run the cells sequentially to:
   - Import necessary libraries
   - Download and load the YOLOv8 model
   - Process the video for player detection
   - Generate the output video with detections

### Code Example

```python
# Import the required libraries
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "football.mp4"
cap = cv2.VideoCapture(video_path)

# Process the video frames
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
```

## 📝 Methodology

1. **Model Selection**: The project uses YOLOv8n, a lightweight version of YOLOv8 that offers a good balance between speed and accuracy.
2. **Preprocessing**: Video frames are extracted and processed for detection.
3. **Detection**: The YOLOv8 model identifies objects in each frame, focusing on the "person" class.
4. **Post-processing**: Bounding boxes are drawn around detected players and the processed frames are compiled into an output video.

## 📊 Results

The model successfully detects football players in the video footage. Detection performance depends on factors such as:
- Lighting conditions
- Camera angle and distance
- Player occlusion
- Video resolution

## 🔮 Future Improvements

- Implement player tracking to maintain player identity across frames
- Add team classification to distinguish between teams
- Include player pose estimation for action recognition
- Optimize for real-time processing on edge devices
- Add player jersey number recognition

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv8 GitHub Repository](https://github.com/ultralytics/ultralytics)
- [Object Detection with YOLOv8](https://learnopencv.com/ultralytics-yolov8/)


## 👏 Acknowledgements

- [Ultralytics](https://ultralytics.com/) for developing the YOLOv8 model
- Open-source computer vision community

---
