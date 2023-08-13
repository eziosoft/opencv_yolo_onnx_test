import os

import cv2

from urls import *
from utils.FPSCounter import FPSCounter
from yolo_onnx.YOLOv7 import YOLOv7_onnx

fps = FPSCounter()

os.environ['DISPLAY'] = ':0.0'
window_name = 'frame'
cap = cv2.VideoCapture(url1)

# Load a model
# onnxModel_path = "models/yolov7/yolov7-tiny.onnx"
# yolo7_onnx = YOLOv7_onnx(onnxModel_path)

onnxModel_path = "models/yolov7/best.onnx"
yolo7_onnx = YOLOv7_onnx(onnxModel_path, classes=["boat", "patio", "people"])

while cap.isOpened():
    success, frame = cap.read()

    if success:
        fps.update()


        detections, ratio, dwdh = yolo7_onnx.detect(frame)
        filter_iou = yolo7_onnx.non_max_suppression(detections, 0.1)
        annotated_frame = yolo7_onnx.drawDetections(filter_iou, frame, ratio, dwdh, filter_classs=None)

        i = 0
        for a in detections:
            if int(a[5]) == 4:
                i += 1

        print(f'FPS = %.2f, people = %d' % (fps.getFPS(), i ), end='\r')
        annotated_frame = cv2.putText(annotated_frame, "Number of people: " + str(i), (50, 50),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
