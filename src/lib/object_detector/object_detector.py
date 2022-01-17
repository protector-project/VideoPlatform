import torch
import cv2
import numpy as np

from models.model import create_model, load_model
from lib.utils.datasets.dataset import letterbox
from utils.general import non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh


class ObjectDetector(object):
    def __init__(self, model_path, device, img_size=1280, stride=32, auto=True):
        print("Creating model...")
        self.model_path = model_path
        self.device = device
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.model = create_model("yolo")
        self.model = load_model(self.model, model_path)
        self.model = self.model.to(device)
        self.model.eval()

    def pre_process(self, im0):
        # Padded resize
        im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.device)
        im = im.float()  # uint8 to fp32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im

    def post_process(self, pred):
        # NMS
        pred = non_max_suppression(pred)
        return pred

    def process_frame(self, im0):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        im = self.pre_process(im0)

        # Inference
        pred = self.model(im)[0]
        pred = self.post_process(pred)

        results = []

        # Process predictions
        for i, det in enumerate(pred):  # per image
            # gn = torch.tensor(im0_shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    results.append((self.cls2label(cls), *xyxy, conf))  # label format

        return results

    def cls2label(self, cls):
        """
        For a given class, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        c = int(cls)  # integer class
        label = self.model.names[c]
        return label

    def count_label(self, results, label):
        n = len([cls == label for cls, xywh, conf in results])
        return n

    def plot_boxes(
        self, results, im0, color=(128, 128, 128), txt_color=(255, 255, 255), lw=3
    ):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        h, w = im0.shape[:2]
        for label, *xyxy, conf in results:
            # xyxy = xywh2xyxy(torch.tensor(xywh).view(1, 4))
            p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            cv2.rectangle(im0, p1, p2, color=color, thickness=lw, lineType=cv2.LINE_AA)
            tf = max(lw - 1, 1)  # font thickness
            # label = self.cls2label(cls)
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[
                0
            ]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(im0, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                im0,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

        return im0
