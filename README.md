## Object Detection Project Overview

This project, developed by Niraj Nepal, implements an object detection system using TensorFlow Lite and OpenCV. It adapts code from a GeeksforGeeks tutorial to demonstrate practical object detection applications with images and live camera feeds.

## Purpose of the Project

The primary goal is to showcase TensorFlow Lite models' capabilities in object detection tasks. This project serves as an educational tool for those new to computer vision and machine learning, illustrating how these technologies can process both static images and real-time video streams.

## Prerequisite Knowledge

Before starting, you should have a basic understanding of:
- Python programming
- TensorFlow and TensorFlow Lite basics
- Neural networks and machine learning concepts
- Basic computer vision concepts using OpenCV

## Methods and Technologies Used

- **TensorFlow Lite**: Manages the loading and execution of the object detection model.
- **OpenCV (cv2)**: Utilized for image and video manipulation.
- **NumPy**: Provides support for high-level mathematical functions on image arrays.
- **Pillow (PIL)**: Helps in manipulating various image file formats.
- **Matplotlib**: Enables creating various types of visualizations in Python.

## AI/ML Topics Covered

- **Object Detection**: The process of identifying and locating objects within images.
- **TensorFlow Lite Models**: Lightweight models designed for efficiency on mobile and edge devices.
- **Real-time Detection**: The ability to process live video streams to detect objects as they appear in real-time.

## Key Terminology

- **Scores**: The confidence levels that the model assigns to its predictions, indicating the probability of each detection being accurate.
- **Classes**: Categories predefined in the model used to classify detected objects.
- **Bounding Boxes**: Rectangles that outline the location of detected objects in an image.
- **Inference**: The process of running a model on data to make predictions.

## Code Structure

- `load_model()`: Loads the TensorFlow Lite model.
- `load_and_prepare_image()`: Prepares images for detection.
- `run_inference()`: Executes the model on an image to perform object detection.
- `draw_detections()`: Annotates images with detection results.
- `display_image()`: Utilizes Matplotlib to display images.
- `detect_live_camera()`: Manages the real-time detection using a live camera feed.
- `main()`: Entry point of the script, handling modes of image or live detection.

## How to Initialize and Use the Project

### Prerequisites Installation

Ensure Python is installed. Install required libraries using pip:

```bash
pip install tensorflow numpy opencv-python pillow matplotlib
```

## Clone the Repository
Clone the GitHub repository or download the Object_Detection_Main.py script to your local machine.

## Prepare the Model and Images
Place the TensorFlow Lite model file in an accessible directory.
If testing with static images, ensure they are also accessible.

## Configuration
Edit Object_Detection_Main.py to set the MODEL_PATH to your model's location.
Set IMAGE_PATH for the image you wish to test.

## Running the Project
Open a terminal or command prompt.
Navigate to the directory containing Object_Detection_Main.py.
Execute the script:
```bash
python Object_Detection_Main.py
```
Follow the on-screen prompts to choose between static image detection (image) and live camera detection (live).
## Conclusion
This project illustrates the practical use of TensorFlow Lite for object detection, enhancing learning through hands-on application and understanding of AI/ML models in real-world scenarios.

