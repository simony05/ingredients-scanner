import numpy as np
import argparse
import imutils
import cv2
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt

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

        # break if image is below minimum size
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # next image
        yield image

def rois_and_locs(pyramid):
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
        for (x, y, roiOriginal) in sliding_window(image, 32, (160, 160)):
            # scale (x, y) of roi in relation to original image dim
            x = int(x * scale)
            w = int(160 * scale)
            h = int(160 * scale)

            # take roi and preprocess for classification
            roi = cv2.resize(roiOriginal, (224, 224))

            # add to list of ROIs
            rois.append(roi)
            locs.append((x, y, x + w, y + h))

    return rois, locs

def predict(rois, locs):
    rois = np.array(rois, dtype = "float32")
    # get predictions, verbose=0 removes loading bar for each prediction
    preds = model.predict(rois, verbose = 0)
    preds = list(zip(preds.argmax(axis = 1).tolist(), preds.max(axis = 1).tolist()))
    labels = {}

    for (i, p) in enumerate(preds):
        (label, prob) = p
        if prob >= 0.75:
            box = locs[i]
            L = labels.get(label, [])
            L.append((box, prob))
            labels[label] = L

    return preds, labels

    #class_idx = np.argmax(prediction)
    #class_label = class_labels[class_idx]
    #return class_label, prediction.flatten()[class_idx], loc

def apply_nms(labels):
    nms_labels = {}
    for label in sorted(labels.keys()):
        boxes = np.array([p[0] for p in labels[label]])
        probs = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, probs)
        nms_labels[label] = boxes.tolist()
    return nms_labels

def show_preds(img, nms_labels):
    class_labels = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber',
                'eggplant', 'garlic', 'ginger', 'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear',
                'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip',
                'watermelon']
    for label in sorted(nms_labels.keys()):
        clone = img.copy()
        boxes = nms_labels[label]
        for (startX, startY, endX, endY) in boxes:
            cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 1)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(clone, str(class_labels[label]), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imshow("labels", clone)
        cv2.waitKey(0)

# load keras model
model = load_model("model.keras")

# load input image
img = cv2.imread('app.jpg')
img = imutils.resize(img, width = 600)
(H, W) = img.shape[:2]

pyramid = image_pyramid(img, scale = 1.5, minSize = (224, 224))
rois, locs = rois_and_locs(pyramid)
preds, labels = predict(rois, locs)
print(preds)
nms_labels = apply_nms(labels)
print(nms_labels.keys())
show_preds(img, nms_labels)
