import os

import cv2
import numpy as np

from urls import *
from utils.FPSCounter import FPSCounter
from yolo_onnx.YOLOv7 import YOLOv7_onnx

fps = FPSCounter()

TILE = True
score_threshold = 0.1

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

# while cap.isOpened():
#     success, frame = cap.read()
#     height, width, _ = frame.shape
#
#     left_half = frame[:, :width // 2, :]
#     right_half = frame[:, width // 2:, :]
# cv2.imshow("left", left_half)
# cv2.imshow("right", right_half)

# if success:
#     fps.update()
#
#     detections1, ratio1, dwdh1 = yolo7_onnx.detect(left_half, 0.1)
#     detections2, ratio2, dwdh2 = yolo7_onnx.detect(right_half, 0.1)
#     # filter_iou = yolo7_onnx.nms(detections, 0.60)
#     annotated_frame1 = yolo7_onnx.drawDetections(detections1, left_half, ratio1, dwdh1, filter_classs=None)
#     annotated_frame2 = yolo7_onnx.drawDetections(detections2, right_half, ratio2, dwdh2, filter_classs=None)
#
#     # i = 0
#     # for a in detections:
#     #     if int(a[5]) == 3:
#     #         i += 1
#
#     # print(f'FPS = %.2f, people = %d' % (fps.getFPS(), i ), end='\r')
#     # print(f"filter_iou: {len(detections)} detections: {len(detections)}\r")
#     # annotated_frame = cv2.putText(annotated_frame, "Number of people: " + str(i), (50, 50),
#     #                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#
#     stitched_frame = np.concatenate((annotated_frame1, annotated_frame2), axis=1)
#     cv2.imshow("stitched", stitched_frame)
#     # cv2.imshow(window_name, annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# else:
#     break

cap.release()
cv2.destroyAllWindows()
