#!/usr/bin/python3

""" Script to train YOLO models on the GOA v1 dataset """

from IPython import display
from IPython.display import Image
import ultralytics
from ultralytics import YOLO
ultralytics.checks()


# Hyperparameters
# model can be set as any of the variants of YOLOv8 or YOLOv9
parameters = {'epochs': 100,
              'imgsz': 320,
              'model': 'models/yolov8s-seg.pt',
              'dataset': 'GOA v1'}

# Load a model
model = YOLO(parameters['model'])
# Train the model
results = model.train(data='datasets/data.yaml', epochs=parameters['epochs'], imgsz=parameters['imgsz'], show_labels=True)
