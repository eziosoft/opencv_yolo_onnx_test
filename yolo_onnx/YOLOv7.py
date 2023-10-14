import random

import cv2
import numpy as np
import onnxruntime as ort


# python3 export.py --grid --end2end --simplify  --topk-all 100  --img-size 640 640 --max-wh 640 --iou-thres 0.2 --conf-thres 0.1  --weights best4.pt
# cp best4.onnx ~/PycharmProjects/opencv_yolo_test/models/yolov7/best.onnx

class YOLOv7_onnx:
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                   'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                   'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                   'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, model_path, classes=class_names):
        self.session = ort.InferenceSession(model_path, providers=['AzureExecutionProvider'
                                                                   ])
        self.names = classes
        self.colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(classes)}

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def detect(self, img, threshold=0.5):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255

        outname = [i.name for i in self.session.get_outputs()]

        inname = [i.name for i in self.session.get_inputs()]

        inp = {inname[0]: im}

        # ONNX inference
        outputs = self.session.run(outname, inp)[0]

        outputs = outputs[outputs[:, 6] > threshold]

        return outputs, ratio, dwdh

    def drawDetections(self, detections, image, ratio, dwdh, filter_classs=None):
        out = image.copy()
        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(detections):
            cls_id = int(cls_id)
            if filter_classs is not None and cls_id not in filter_classs:
                continue

            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()

            score = round(float(score), 3)
            name = self.names[cls_id]
            color = self.colors[name]
            name += ' ' + str(score)
            cv2.rectangle(out, box[:2], box[2:], color, 2)
            cv2.putText(out, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
        return out

    def nms(self, detections, iou_threshold=0.5):
        scores = [detection[6] for detection in detections]
        boxes = [detection[1:5] for detection in detections]

        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)

        filtered_detections = [detections[i] for i in indices]

        return filtered_detections
