import cv2
import numpy as np
from ultralytics import YOLO
from keras.models import load_model

# Load the YOLOv8 object detection model
yolo_model = YOLO("yolov8s.pt")

# Load the custom image recognition model
image_recognition_model = load_model("model.keras")

# Load the image
img = cv2.imread('apple.jpg')

# Perform object detection using YOLOv8
results = yolo_model(img)

# Loop through the detected objects
for result in results:
    print(result.boxes)
    result.show()
    # Get the bounding box coordinates
    px = res.xyxy

    # Crop the object from the original image
    object_img = img[int(y):int(y2), int(x):int(x2)]

    # Preprocess the object image for the image recognition model
    object_img = cv2.resize(object_img, (224, 224))
    object_img = object_img / 255.0
    object_img = np.expand_dims(object_img, axis=0)

    # Run the object image through the image recognition model
    predictions = image_recognition_model.predict(object_img)

    # Get the top prediction
    top_pred = np.argmax(predictions)

    class_labels = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber',
                'eggplant', 'garlic', 'ginger', 'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 'ango', 'onion', 'orange', 'paprika', 'pear',
                'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'oy beans', 'pinach', 'weetcorn', 'weetpotato', 'tomato', 'turnip',
                'watermelon']
    
    # Print the class label and confidence
    print(f"Object detected: {class_labels[top_pred]} with confidence {predictions[top_pred]}")