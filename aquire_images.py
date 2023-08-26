import datetime

from roboflow import Roboflow
import cv2

from secret import roboflow_api_key
from urls import url1, url2

# Initialize the Roboflow object with your API key
rf = Roboflow(api_key=roboflow_api_key)

# Retrieve your current workspace and project name
print(rf.workspace())

# Specify the project for upload
project = rf.workspace("test-workspace-a6u9w").project("boats-gpvac")

window_name = 'frame'
cap = cv2.VideoCapture(url2)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            cv2.imwrite("./img/" + time + ".jpg", frame)
            project.upload("./img/" + time + ".jpg")
            print("Uploaded")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
