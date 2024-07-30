import cv2
import numpy as np
from keras.models import load_model

# load image
img = cv2.imread('apple.jpg')
img = cv2.resize(img, (600, 600))

# sliding window parameters
win_size = (600, 600)
step_size = (int(win_size[0] / 2), int(win_size[1] / 2))

# load keras model
model = load_model("model.keras")

# detected objects
objects = []
exists = set()

# class names
class_labels = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber',
                'eggplant', 'garlic', 'ginger', 'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 'ango', 'onion', 'orange', 'paprika', 'pear',
                'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'oy beans', 'pinach', 'weetcorn', 'weetpotato', 'tomato', 'turnip',
                'watermelon']

# iterate over sliding window
while (win_size[0] >= 224 and win_size[1] >= 224):
    for x in range(0, img.shape[1] - win_size[0] + 1, step_size[0]):
        for y in range(0, img.shape[0] - win_size[1] + 1, step_size[1]):
            # current window
            window = img[y:y+win_size[1], x:x+win_size[0], :]

            # preprocess window
            window = cv2.resize(window, (224, 224))
            window = window / 255.0
            window = np.expand_dims(window, axis=0)  

            # classify window
            predictions = model.predict(window, verbose = 0)
            predictions = np.squeeze(predictions)

            # top prediction
            top_pred = np.argmax(predictions)

            if predictions[top_pred] > 0.90:
                # Add the object to the list
                if class_labels[top_pred] not in exists:
                    objects.append((x, y, win_size[0], win_size[1], class_labels[top_pred]))
                    exists.add(class_labels[top_pred])

    win_size = (int(win_size[0] / 1.5), int(win_size[1] / 1.5))
    step_size = (int(step_size[0] / 1.5), int(step_size[1] / 1.5))

# Draw the detected objects on the image
for obj in objects:
    cv2.rectangle(img, (obj[0], obj[1]), (obj[0]+obj[2], obj[1]+obj[3]), (0, 255, 0), 2)
    y = obj[1] - 10 if obj[1] - 10 > 10 else obj[1] + 10
    cv2.putText(img, obj[4], (obj[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()