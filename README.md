# Object Detection System

This repository hosts the code and necessary resources for a dynamic object detection system capable of categorizing objects within video streams.

## Project Overview

The main goal of this project is to create an object detection system that can analyze video streams and identify objects in real-time. Key features and components of the project include:

- Integration of the SSD architecture for real-time object detection.
- Utilization of OpenCV's DNN module for seamless model incorporation and inference.
- Analysis of convolutional layers to generate class IDs and bounding box predictions.
- Confidence thresholding to filter out low-confidence detections and select the most probable results.
- Implementation of Non-Maximum Suppression (NMS) to handle overlapping bounding boxes.
- Color-coded visualization using the numpy library for intuitive object categorization.
- Annotated bounding boxes to aid accurate object identification.
- Dynamic calculation and display of Frames Per Second (FPS) for insights into real-time processing efficiency.

## Repository Contents

This repository contains the following files and directories:

Files:
1. `coco.names`: Contains the class labels for the COCO dataset.
2. `frozen_inference_graph.pb`: The frozen inference graph of the trained object detection model.
3. `main.py`: The main Python script that implements the object detection system and serves as an entry point for running the project.
4. `objectdetector.py`: The Python script containing the `ObjectDetector` class responsible for object detection operations.
5. `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`: The configuration file for the pre-trained SSD MobileNet V3 Large model.

## Requirements

Ensure you have the following dependencies installed:

- Python
- OpenCV (cv2)
- numpy
