import cv2
import numpy as np
from keras.models import load_model

# load image
img = cv2.imread('apple.jpg')
img = cv2.resize(img, (600, 600))

# sliding window parameters
win_size = (img.shape[0], img.shape[1])
step_size = (10, 10)

# load keras model
model = load_model("model.keras")

# detected objects
objects = []

# iterate over sliding window
while (win_size[0] >= 224):
    for x in range(0, img.shape[1], step_size[0]):
        for y in range(0, img.shape[0], step_size[1]):
            # current window
            window = img[y:y+win_size[1], x:x+win_size[0], :]

            # Preprocess the window
            window = cv2.resize(window, (224, 224))
            window = window / 255.0
            window = np.expand_dims(window, axis=0)

            # Classify the window
            predictions = model.predict(window)

            # Get the top prediction
            top_pred = np.argmax(predictions)

            # Check if the prediction is above a certain threshold
            if predictions[top_pred] > 0.5:
                # Add the object to the list
                objects.append((x, y, win_size[0], win_size[1]))
    win_size = (int(win_size[0] / 1.5), int(win_size[1] / 1.5))

# Draw the detected objects on the image
for obj in objects:
    cv2.rectangle(img, (obj[0], obj[1]), (obj[0]+obj[2], obj[1]+obj[3]), (0, 255, 0), 2)

# Display the image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()