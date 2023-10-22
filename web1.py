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
cap = cv2.VideoCapture(url1)
onnxModel_path = "models/yolov7/best.onnx"
yolo7_onnx = YOLOv7_onnx(onnxModel_path, classes=["boat", "crowd", "patio", "person"])


def generate_frames():
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            # If no frame is available, create a black frame with text and yield it
            height, width, _ = frame.shape
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            text = "No video available"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            cv2.putText(black_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', black_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:

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

                people = count_people(detections)
                annotated_frame = cv2.putText(annotated_frame, "Number of people: " + str(people), (50, 50),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def annotate_square(square):
    detections, ratio, dwdh = yolo7_onnx.detect(square, score_threshold)
    annotated_square = yolo7_onnx.drawDetections(detections, square, ratio, dwdh, filter_classs=None)
    return annotated_square


def count_people(detections):
    i = 0
    for a in detections:
        if int(a[5]) == 3:
            i += 1
    return i

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)

cap.release()
cv2.destroyAllWindows()
