import os

import cv2

from urls import *
from utils.FPSCounter import FPSCounter
from yolo_onnx.YOLOv7 import YOLOv7_onnx

fps = FPSCounter()

os.environ['DISPLAY'] = ':0.0'
window_name = 'frame'
cap = cv2.VideoCapture(0)

# Load a model
onnxModel_path = "models/yolov7/yolov7-tiny.onnx"
yolo7_onnx = YOLOv7_onnx(onnxModel_path)

onnxModel_path = "models/yolov7/best.onnx"
yolo7_onnx = YOLOv7_onnx(onnxModel_path, classes=["boat", "crowd", "patio", "person"])

while cap.isOpened():
    success, frame = cap.read()

    if success:
        fps.update()

        detections, ratio, dwdh = yolo7_onnx.detect(frame, 0.4)
        # filter_iou = yolo7_onnx.nms(detections, 0.60)
        annotated_frame = yolo7_onnx.drawDetections(detections, frame, ratio, dwdh, filter_classs=None)

        i = 0
        for a in detections:
            if int(a[5]) == 3:
                i += 1

        # print(f'FPS = %.2f, people = %d' % (fps.getFPS(), i ), end='\r')
        # print(f"filter_iou: {len(detections)} detections: {len(detections)}\r")
        annotated_frame = cv2.putText(annotated_frame, "Number of people: " + str(i), (50, 50),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
