#!/usr/bin/python3

""" Script to evaluate the trained Mask R-CNN model on the test data"""

# Import necessary libraries
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo


# Function to run inference on an image
def infer_on_image(image_path):
    """
    Run inferences on an image and visualize the results.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        int: Number of detected instances.

    Description:
        This function loads an image from the given path, runs the predictor model on it,
        and visualizes the results by drawing segmentation masks and bounding boxes.
        The output image is then saved to a directory.

    Notes:
        - The predictor model is assumed to be already loaded and configured.
        - The visualization colors can be customized by modifying the `color` arguments.
    """
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    outputs = predictor(im)

    # Count the number of detected instances
    num_instances = len(outputs["instances"])
    
    # Remove class labels and confidence scores
    instances = outputs["instances"].to("cpu")
    instances.remove("pred_classes")
    instances.remove("scores")
    
    # Visualize the remaining instances (only segmentation masks)
    v = Visualizer(im[:, :, ::-1], scale=1)
    
    # Set all masks to color of your choice
    masks = instances.pred_masks
    for mask in masks:
        mask_np = mask.numpy()
        v.draw_binary_mask(mask_np, color="green", alpha=0.4)

    # Draw bounding boxes
    boxes = instances.pred_boxes.tensor.numpy()
    for box in boxes:
        v.draw_box(box, edge_color="green")
    
    # Convert the result to an image and display/save it
    im = Image.fromarray(v.output.get_image()[:, :, ::-1])
    # im.show()

    # Save the image with inference results
    save_dir = "./inference_result"
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    im.save(save_path, quality=95, dpi=(300, 300))
    print(f"Results have been saved at: {save_dir}")

    return num_instances


# Register datasets (if not already registered)
register_coco_instances('goa_test', {}, f'dataset/test/_annotations.coco.json', f'dataset/test/')

# Set up configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join("./goa_output/mask_rcnn_R_50_FPN_3x/2024-05-28-01-13-10/model_final.pth")  # path to the trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.INPUT.MASK_FORMAT = 'polygon'

# Create a predictor
predictor = DefaultPredictor(cfg)

# Load the dataset
dataset_dicts = DatasetCatalog.get("goa_test")

# Run inference on a random sample of test images
for d in random.sample(dataset_dicts, 3):
    infer_on_image(d["file_name"])
