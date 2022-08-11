import cv2
import numpy as np
import torch
import gdown
from os.path import exists as file_exists
from lib.utils.general import xyxy2xywh

from models.model import create_model, load_model


class Tracker(object):
    def __init__(self, opt, device):
        self.opt = opt
        opt.device = device
        print('Creating model...')
        self.model = create_model(opt)
        # self.model = load_model(self.model, opt)
        # self.model = self.model.to(device)
        # self.model.eval()

    def process_frame(self, im0, prev_frame, dets):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        outputs = []
        
        if self.opt.ecc:  # camera motion compensation
            self.model.tracker.camera_update(prev_frame, im0)

        if dets is not None and len(dets):
            xywhs = xyxy2xywh(dets[:, 0:4])
            confs = dets[:, 4]
            clss = dets[:, 5]
            outputs = self.model.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
        else:
            self.model.increment_ages()

        return outputs