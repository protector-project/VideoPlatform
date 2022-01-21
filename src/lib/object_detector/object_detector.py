import torch
import cv2
import numpy as np

from models.model import create_model, load_model
from datasets.dataset import letterbox
from utils.general import non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh


class ObjectDetector(object):
    def __init__(self, opt, device):
        print("Creating model...")
        self.opt = opt
        self.device = device
        self.model = create_model("yolo")
        self.model = load_model(self.model, opt.detection_model)
        self.model = self.model.to(device)
        self.model.eval()

        self.imgsz = self.opt.detection_imgsz
        self.stride = int(self.model.stride.max())

    def pre_process(self, im0):
        # Padded resize
        im = letterbox(im0, self.imgsz, stride=self.stride)[0]

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
        im = self.pre_process(im0.copy())

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
        n = len([cls == label for cls, *xywh, conf in results])
        return n
