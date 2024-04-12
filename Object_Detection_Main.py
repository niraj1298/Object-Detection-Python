#
# Developer and Project Information:
#
# Name: Object Detection 
# Developed by: Niraj Nepal
# Code Sourced Citations: https://www.geeksforgeeks.org/object-detection-using-tensorflow/ (Modified this code for Object_Detection_Main.py)
# Date: April 19, 2024
#

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
import sys
import time


# Configuration section with paths and label information for ML Model and Image being test

MODEL_PATH = "C:/Users/pokem/OneDrive/Desktop/Python stuff/AI-Object-Detection/models/custom_model_lite/detect.tflite"
IMAGE_PATH = "C:/Users/pokem/OneDrive/Desktop/Python stuff/AI-Object-Detection/Images/image_test_10.png"
LABELS = [None] * 10 + ["person"]


def load_model(tflite_model_path):
    """
    Load the TensorFlow Lite model from the specified path and allocate tensors.

    This function initializes the TensorFlow Lite interpreter with the model
    specified by the `tflite_model_path`. It then allocates tensors for the model,
    preparing it for inference. This is a necessary step before performing any
    predictions with the model.

    Parameters:
    - tflite_model_path (str): The file path to the TensorFlow Lite model file.

    Returns:
    - interpreter (tf.lite.Interpreter): The loaded and initialized TensorFlow Lite
      interpreter object, ready for performing inference.

    Usage:
    - To use this function, provide the path to your TensorFlow Lite model file.
      The function will return an interpreter object which can be used to perform
      inference with the model.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter


def load_and_prepare_image(image_path, model_input_shape=(320, 320)):
    """
    Load an image from the specified path, convert it to RGB, resize it to the specified model input shape,
    normalize its pixel values to the range [0, 1], and convert it into a tensor suitable for model input.

    This function is designed to prepare images for inference with a TensorFlow model. The image is first
    checked for existence at the given path. It is then opened, converted to RGB (if not already in that format),
    and resized to match the input shape expected by the model. The pixel values are normalized to fall within
    the range [0, 1], which is a common requirement for neural network inputs. Finally, the image is converted
    to a TensorFlow tensor, and a batch dimension is added to make the image tensor compatible with the model's
    expected input format.

    Parameters:
    - image_path (str): The file path to the image to be loaded and prepared.
    - model_input_shape (tuple of int, optional): The height and width to which the image should be resized,
      specified as (height, width). Defaults to (320, 320).

    Returns:
    - tf.Tensor: A tensor representing the processed image, ready to be fed into a TensorFlow model for inference.

    Usage:
    - This function can be used to prepare images for object detection or classification models that require
      input images to be in a specific format. It simplifies the process of image preprocessing for inference.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path {image_path} does not exist.")
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize(model_input_shape)
    image_np = np.array(image)
    image_np = image_np / 255.0 
    image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32) 
    image_tensor = tf.expand_dims(image_tensor, 0) 
    return image_tensor


def run_inference(interpreter, image_tensor):
    """
    Perform inference on an input image tensor using the provided TensorFlow Lite interpreter.

    This function sets the input tensor of the interpreter to the provided image tensor, invokes the interpreter
    to perform inference, and then collects the output tensors. These outputs typically include detection boxes,
    scores, and classes for object detection models. Additionally, the number of detections is also retrieved.
    The shapes of the output tensors are printed for debugging purposes.

    Parameters:
    - interpreter (tf.lite.Interpreter): The TensorFlow Lite interpreter initialized with a model.
    - image_tensor (tf.Tensor): The input image tensor to run inference on.

    Returns:
    - dict: A dictionary containing the inference results, including detection boxes, scores, classes,
      and the number of detections.

    Usage:
    - This function is intended to be used with TensorFlow Lite models that perform object detection.
      It requires a pre-loaded and initialized interpreter and an image tensor prepared according to the
      model's expected input format. The function facilitates the process of running inference and extracting
      the results for further processing or analysis.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image_tensor.numpy())
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[2]['index'])
    print(f"Boxes shape: {boxes.shape}, Scores shape: {scores.shape}, Classes shape: {classes.shape}")

    detections = {
        'detection_boxes': boxes,  
        'detection_scores': scores,
        'detection_classes': classes,
        'num_detections': interpreter.get_tensor(output_details[3]['index'])      
    }
    return detections


def draw_detections(image_np, detections, labels, threshold=0.5):
    """
    Draws detection boxes on an image with labels and scores for each detected object.

    This function iterates over all detections and draws a rectangle around the detected objects
    that have a score above the specified threshold. It also annotates the rectangle with the
    object's label and score. The function modifies the input image in place and returns the
    modified image.

    Parameters:
    - image_np (numpy.ndarray): The input image as a NumPy array.
    - detections (dict): A dictionary containing the detection results. The dictionary should
      include keys 'detection_boxes', 'detection_scores', and 'detection_classes', which
      correspond to the bounding boxes, scores, and class indices of the detected objects,
      respectively.
    - labels (list): A list of strings representing the labels for each class index. The index
      in this list should correspond to the class index in the detections.
    - threshold (float, optional): The score threshold for displaying detections. Only detections
      with a score above this threshold will be drawn. Defaults to 0.5.

    Returns:
    - numpy.ndarray: The input image with detection boxes, labels, and scores drawn on it.

    Usage:
    This function is intended to be used after running object detection on an image to visually
    display the detection results. It requires the original image, the detection results, and
    a list of labels for the detected classes. The function is useful for debugging and
    visualization purposes.
    """
    boxes = detections['detection_boxes'][0] 
    scores = detections['detection_scores'][0]  
    classes = detections['detection_classes'][0]  

    for i in range(scores.shape[0]):
        score = scores[i]
        if score > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * image_np.shape[1], xmax * image_np.shape[1],
                                          ymin * image_np.shape[0], ymax * image_np.shape[0])
            cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            class_index = int(classes[i]) if classes.ndim > 0 else int(classes)
            label = labels[class_index] if class_index < len(labels) else "Unknown"
            label_text = f"{label}: {score:.2f}"

            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            text_x = int(left) + 20  
            text_y = int(bottom) - 20  

            cv2.rectangle(image_np, (text_x, text_y - text_height - 2), (text_x + text_width, text_y), (0, 255, 0), -1)
            cv2.putText(image_np, label_text, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image_np

def display_image(image_np):
    """
    Displays an image using Matplotlib.

    This function takes an image represented as a NumPy array and displays it using Matplotlib's
    imshow function. The axis is turned off to ensure that only the image is displayed without any
    additional chart information. This function is typically used for visualizing images in
    notebooks or Python scripts where inline image display is required.

    Parameters:
    - image_np (np.array): Image to display as a NumPy array.
    """
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()

def detect_live_camera(model, labels):
    """
    Detects objects in real-time using a live camera feed.

    This function initializes a camera feed, performs object detection on each frame in real-time,
    and displays the annotated frames with detection results. It uses a pre-trained model for
    detection and annotates each frame with bounding boxes, class labels, and detection scores.
    The function also calculates and displays the frames per second (FPS) to provide insight into
    the performance of the object detection process in real-time. The camera feed can be stopped
    by pressing 'q'.

    Parameters:
    - model: The pre-trained object detection model.
    - labels (list): A list of strings representing the labels for each class index detected by the model.

    How this will be used:
    This function is intended to be used in applications requiring real-time object detection,
    such as surveillance systems, robotics, or interactive installations. It provides a simple
    interface for integrating object detection into live video streams.
    """
    camera_index = input("Enter the camera index (default is 0): ")
    camera_index = int(camera_index) if camera_index.isdigit() else 0

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FPS, 60) 

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        sys.exit(1)

    fps_counter = 0
    fps_timer = time.time()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            fps_counter += 1
            if time.time() - fps_timer > 1:
                fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()

            image_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.uint8)
            detections = run_inference(model, image_tensor)
            annotated_image = draw_detections(frame, detections, labels)

            cv2.putText(annotated_image, f"FPS: {fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Live Object Detection', annotated_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)


def main():
    """
    Main function to execute the object detection workflow.
    
    This function serves as the entry point for the object detection application. It allows the user to choose between
    image detection and live camera detection modes. Based on the user's choice, it either processes a single image
    and displays the result with detected objects annotated, or it starts a live detection session using the camera feed.
    
    The choice between 'image' and 'live' mode dictates the flow of the application:
    - In 'image' mode, the application loads a predefined image, prepares it for detection, runs the detection model,
      annotates the detected objects on the image, and finally displays the annotated image.
    - In 'live' mode, the application continuously captures video frames from the camera, processes each frame for
      object detection, annotates the frames with detected objects, and displays the live video with annotations in real-time.
    
    This flexibility allows the application to be used in various scenarios, such as processing pre-captured images for
    analysis or performing real-time object detection in scenarios like surveillance, robotics, or interactive installations.
    """
    model = load_model(MODEL_PATH)
    mode = input("Enter 'image' for image detection or 'live' for live camera detection: ").strip().lower()
    
    if mode == 'image':
        image_tensor = load_and_prepare_image(IMAGE_PATH, model_input_shape=(320, 320))
        detections = run_inference(model, image_tensor)
        image_np = np.squeeze(image_tensor.numpy())
        annotated_image = draw_detections(image_np, detections, LABELS)
        display_image(annotated_image)
    elif mode == 'live':
        detect_live_camera(model, LABELS)
    else:
        print("Invalid mode selected. Please choose 'image' or 'live'.")

if __name__ == "__main__":
    main()