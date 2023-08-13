import datetime
import cv2
from roboflow import Roboflow

from secret import roboflow_api_key
from yolo_onnx.YOLOv7 import YOLOv7_onnx


DETECT = False
if not DETECT:
    # Initialize the Roboflow object with your API key
    rf = Roboflow(api_key=roboflow_api_key)

    # Retrieve your current workspace and project name
    print(rf.workspace())

    # Specify the project for upload
    project = rf.workspace("test-workspace-a6u9w").project("boats-gpvac")

url1 = "https://deliverys6.quanteec.com/contents/encodings/live/4956e420-26db-4c3d-3538-3330-6d61-63-8c5b-4ef84c215770d/master.m3u8"
url2 = "https://deliverys6.quanteec.com/contents/encodings/live/cc6af81c-bdf7-4ae8-3438-3330-6d61-63-bf1d-477d57f12531d/master.m3u8"
url3 = "https://deliverys5.quanteec.com/contents/encodings/live/e6b6f1cd-ae5d-40f2-3032-3630-6d61-63-a744-dbccb109af26d/media_0.m3u8"

window_name = 'frame'
cap = cv2.VideoCapture(url1)

# Load a model
# onnxModel_path = "models/yolov7-tiny.onnx"
# yolo7_onnx = YOLOv7_onnx(onnxModel_path)

onnxModel_path = "models/yolov7/best4.onnx"
yolo7_onnx = YOLOv7_onnx(onnxModel_path, classes=["boat", "patio", "people"])

while cap.isOpened():
    success, frame = cap.read()

    if success:
        if DETECT:
            detections, ratio, dwdh = yolo7_onnx.detect(frame)
            filter_iou = yolo7_onnx.non_max_suppression(detections, 0.1)
            annotated_frame = yolo7_onnx.drawDetections(filter_iou, frame, ratio, dwdh, filter_classs=None)

            i = 0
            for a in detections:
                print(a)
                if int(a[5]) == 3:
                    i += 1

            annotated_frame = cv2.putText(annotated_frame, "Number of people: " + str(i), (50, 50),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, annotated_frame)

        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                cv2.imwrite("./img/" + time + ".jpg", frame)
                project.upload("./img/" + time + ".jpg")
                print("Uploaded")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
