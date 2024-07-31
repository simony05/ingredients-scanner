from detecto import core, utils, visualize
import cv2

image = cv2.imread("cucmber.jpeg")
#Initializes a pre-trained model
model = core.Model()
labels, boxes, scores = model.predict_top(image)
visualize.show_labeled_image(image, boxes, labels)