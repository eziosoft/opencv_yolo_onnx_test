import os

import cv2
import numpy as np

from urls import *
from utils.FPSCounter import FPSCounter
from yolo_onnx.YOLOv7 import YOLOv7_onnx

fps = FPSCounter()

TILE = False
score_threshold = 0.2

os.environ['DISPLAY'] = ':0.0'
window_name = 'frame'
cap = cv2.VideoCapture(url1)

# Load a model
# onnxModel_path = "models/yolov7/yolov7-tiny.onnx"
# yolo7_onnx = YOLOv7_onnx(onnxModel_path)

onnxModel_path = "models/yolov7/best.onnx"
yolo7_onnx = YOLOv7_onnx(onnxModel_path, classes=["boat", "crowd", "patio", "person"])


def annotate_square(square):
    # Replace this with your annotation logic
    detections, ratio, dwdh = yolo7_onnx.detect(square, score_threshold)
    annotated_square = yolo7_onnx.drawDetections(detections, square, ratio, dwdh, filter_classs=None)
    return annotated_square


while cap.isOpened():
    # Assuming you've already captured a frame using cap.read()
    success, frame = cap.read()

    # Get frame dimensions
    height, width, _ = frame.shape

    if TILE:
        # Define the size of the squares
        square_size = 640

        # Initialize lists to store annotated squares
        annotated_squares = []

        # Split the frame into 640x640 squares, apply annotation, and store in list
        for y in range(0, height, square_size):
            for x in range(0, width, square_size):
                square = frame[y:y + square_size, x:x + square_size]
                annotated_square = annotate_square(square)
                annotated_squares.append(annotated_square)

        # Stitch annotated squares back together
        rows = []
        for i in range(0, len(annotated_squares), width // square_size):
            row = np.hstack(annotated_squares[i:i + width // square_size])
            rows.append(row)

        annotated_frame = np.vstack(rows)
    else:
        detections, ratio, dwdh = yolo7_onnx.detect(frame, score_threshold)
        annotated_frame = yolo7_onnx.drawDetections(detections, frame, ratio, dwdh, filter_classs=None)

    cv2.imshow(window_name, annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
