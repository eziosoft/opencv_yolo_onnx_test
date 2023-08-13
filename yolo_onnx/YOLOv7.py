import random

import cv2
import numpy as np
import onnxruntime as ort


# python3 export.py --weights yolov7-tiny.pt --grid --end2end --simplify \
#         --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640

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

    def __init__(self, model_path, cuda=False, classes=class_names):
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=self.providers)
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

    def detect(self, img):
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

    def calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        intersection_x1 = max(x1, x2)
        intersection_y1 = max(y1, y2)
        intersection_x2 = min(x1 + w1, x2 + w2)
        intersection_y2 = min(y1 + h1, y2 + h2)

        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)

        intersection_area = intersection_width * intersection_height

        area_box1 = w1 * h1
        area_box2 = w2 * h2

        iou = intersection_area / (area_box1 + area_box2 - intersection_area)
        return iou

    def non_max_suppression(self, detections, threshold):
        boxes = [d[1:5] for d in detections]
        scores = [d[6] for d in detections]

        sorted_indices = np.argsort(scores)[::-1]
        selected_indices = []

        while len(sorted_indices) > 0:
            best_index = sorted_indices[0]
            selected_indices.append(best_index)

            remaining_indices = sorted_indices[1:]
            to_delete = []

            for idx in remaining_indices:
                iou = self.calculate_iou(boxes[best_index], boxes[idx])
                if iou >= threshold:
                    to_delete.append(idx)

            # Ensure indices to delete are within valid range
            to_delete = [idx for idx in to_delete if idx < len(sorted_indices)]

            sorted_indices = np.delete(sorted_indices, [0] + to_delete)

        selected_detections = [detections[idx] for idx in selected_indices]
        return selected_detections

