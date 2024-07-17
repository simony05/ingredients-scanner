import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# load keras model
model = load_model("model.keras")

# load image
img = load_img('apple.jpg', target_size=(224, 224))

# prepare image
x = img_to_array(img);
x = x / 255.0
x = np.expand_dims(x, axis = 0)

# make predictions on image
predictions = model.predict(x)

# get class label with highest probability
class_idx = np.argmax(predictions)
class_labels = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber',
                'eggplant', 'garlic', 'ginger', 'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear',
                'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip',
                'watermelon']
class_label = class_labels[class_idx]

print(predictions)
print(f"Predicted class: {class_label}")