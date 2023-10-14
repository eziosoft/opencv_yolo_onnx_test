from flask import Flask, Response, render_template
import os
import cv2
from urls import *
from utils.FPSCounter import FPSCounter
from yolo_onnx.YOLOv7 import YOLOv7_onnx

app = Flask(__name__)

# Initialize global variables
fps = FPSCounter()
os.environ['DISPLAY'] = ':0.0'
cap = cv2.VideoCapture(url1)
onnxModel_path = "models/yolov7/best.onnx"
yolo7_onnx = YOLOv7_onnx(onnxModel_path, classes=["boat", "crowd", "patio", "person"])


def generate_frames():
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            fps.update()

            detections, ratio, dwdh = yolo7_onnx.detect(frame, 0.2)
            filter_iou = yolo7_onnx.nms(detections, 0.90)

            print(f"filter_iou: {len(filter_iou)} detections: {len(detections)}\r")
            annotated_frame = yolo7_onnx.drawDetections(filter_iou, frame, ratio, dwdh, filter_classs=None)

            i = 0
            for a in filter_iou:
                if int(a[5]) == 2:
                    i += 1

            # print(f'FPS = %.2f, people = %d' % (fps.getFPS(), i ), end='\r')
            annotated_frame = cv2.putText(annotated_frame, "Number of people: " + str(i), (50, 50),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        else:
            break


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
