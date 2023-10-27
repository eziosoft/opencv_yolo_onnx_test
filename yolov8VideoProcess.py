import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from urls import url1

model = YOLO('yolov8n.pt')
annotator = sv.BoxCornerAnnotator(thickness=2, corner_length=5)

video_path = url1
video_out = 'out.mp4'
start_frame = 23 * 60 * 7
classes_to_detect = [0, 2, 7]
slice_size = 640

video_info = sv.VideoInfo.from_video_path(video_path=video_path)


def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model.predict(image_slice, classes=classes_to_detect)[0]
    return sv.Detections.from_ultralytics(result)


slicer = sv.InferenceSlicer(callback=callback, slice_wh=(slice_size, slice_size))

with sv.VideoSink(target_path=video_out, video_info=video_info, codec='H264') as sink:
    for frame in sv.get_video_frames_generator(source_path=video_path, start=start_frame, stride=5):
        detections = slicer(frame)
        annotated_frame = annotator.annotate(scene=frame.copy(), detections=detections)

        people = detections[detections.class_id == 0]
        cars = detections[detections.class_id == 2]
        trucks = detections[detections.class_id == 7]

        cv2.putText(annotated_frame, "people=%d" % (len(people)), (40, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, "cars=%d" % (len(cars) + len(trucks)), (40, 120), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('image window', annotated_frame)
        # sink.write_frame(annotated_frame)
        cv2.waitKey(1)




