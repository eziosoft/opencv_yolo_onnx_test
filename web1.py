import os
import cv2
import numpy as np
from flask import Flask, Response
from urls import *
from utils.FPSCounter import FPSCounter
from yolo_onnx.YOLOv7 import YOLOv7_onnx

app = Flask(__name__)

fps = FPSCounter()
TILE = False
score_threshold = 0.2
# os.environ['DISPLAY'] = ':0.0'
cap = cv2.VideoCapture(url2)
onnxModel_path = "models/yolov7/best.onnx"
yolo7_onnx = YOLOv7_onnx(onnxModel_path, classes=["boat", "crowd", "patio", "person"])


def generate_frames():
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        height, width, _ = frame.shape

        if TILE:
            square_size = 640
            annotated_squares = []

            for y in range(0, height, square_size):
                for x in range(0, width, square_size):
                    square = frame[y:y + square_size, x:x + square_size]
                    annotated_square = annotate_square(square)
                    annotated_squares.append(annotated_square)

            rows = []
            for i in range(0, len(annotated_squares), width // square_size):
                row = np.hstack(annotated_squares[i:i + width // square_size])
                rows.append(row)

            annotated_frame = np.vstack(rows)
        else:
            detections, ratio, dwdh = yolo7_onnx.detect(frame, score_threshold)
            annotated_frame = yolo7_onnx.drawDetections(detections, frame, ratio, dwdh, filter_classs=None)

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def annotate_square(square):
    detections, ratio, dwdh = yolo7_onnx.detect(square, score_threshold)
    annotated_square = yolo7_onnx.drawDetections(detections, square, ratio, dwdh, filter_classs=None)
    return annotated_square


@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)

cap.release()
cv2.destroyAllWindows()
