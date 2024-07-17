import imutils
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# input image, step size, window size
def sliding_window(image, step, ws):
    # loop over rows
    for y in range(0, image.shape[0] - ws[1], step):
        # loop over columns
        for x in range(0, image.shape[1] - ws[0], step):
            # window of current image
            yield(x, y, image[y:y + ws[1], x:x + ws[0]])

# input image, scale factor (larger = fewer layers), minimum size of image
# loops from original image size to until less than min size
def image_pyramid(image, scale = 1.5, minSize = (224, 224)):
    # original image
    yield image

    # keep looping over image pyramid
    while True:
        # compute dimensions of next image in pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width = w)

        # break if image is below minimum sie=ze
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # next image
        yield image

# width: 600, scale: 1.5, window step: 16, roi size: (200, 150), input: (224, 224)

# load keras model
model = load_model("model.keras")

# load input image
orig = cv2.imread('apple.jpg')
orig = imutils.resize(orig, width = 600)
(H, W) = orig.shape[:2]

# initialize image pyramid
pyramid = image_pyramid(orig, scale = 1.5, minSize = (200, 150))

# initialize lists
    # ROIs generated from image pyramid and sliding window
    # (x,y) coordinates of where ROI is in original image
rois = []
locs = []

# loop over image pyramid
for image in pyramid:
    # determine scale factor between original image and current layer of pyramid
    scale = W / float(image.shape[1])

    # for each layer of image pyramid, loop over sliding window
    for (x, y, roiOriginal) in sliding_window(image, 16, (200, 150)):
        # scale (x, y) of roi in relation to original image dim
        x = int(x * scale)
        y = int(y * scale)
        w = int(224 * scale)
        h = int(224 * scale)

        # take roi and preprocess for classification
        roi = cv2.resize(roiOriginal, (224, 224))
        roi = img_to_array(roi)
        roi = roi / 255.0
        roi = np.expand_dims(roi, axis = 0)

        # add to list of ROIs
        rois.append(roi)
        locs.append((x, y, x + w, y + h))

def predict(roi, loc):
    # get predictions, verbose=0 removes loading bar for each prediction
    prediction = model.predict(roi, verbose = 0)
    # get class label with highest probability
    class_idx = np.argmax(prediction)
    class_labels = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber',
                'eggplant', 'garlic', 'ginger', 'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear',
                'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip',
                'watermelon']
    class_label = class_labels[class_idx]
    return class_label, prediction.flatten()[class_idx], loc

# classify each roi
labels = set()
results = []
for i, r in enumerate(rois):
    pred = predict(rois[i], locs[i])
    if pred[0] not in labels and pred[1] > 0.5:
        labels.add(pred[0])
        results.append(pred)
        subimg = orig[pred[2][1]:pred[2][3], pred[2][0]:pred[2][2]]
        cv2.imshow(pred[0], subimg)

print(results)