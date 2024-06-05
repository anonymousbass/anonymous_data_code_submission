#!/usr/bin/python3

""" Script to evaluate the trained YOLO models on the GOA v1 dataset """

from IPython import display
from IPython.display import Image
import ultralytics
from ultralytics import YOLO
ultralytics.checks()


# Change the path to the correct path for the desired model after training the model
model_path = 'runs/segment/train1/weights/best.pt'
trained_model = YOLO(model_path)
test_model = trained_model.predict(source='datasets/test/images', conf=0.7, 
                                   imgsz=320, show_labels=False, save=True)
