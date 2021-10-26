import itertools
import math
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from vidgear.gears import CamGear

from model.base_model import *


class AnomalyDetector:
    def __init__(self):
        self.MPN_model_dir = (
            "pre_trained_models/mpn_piazza_2_sett_3_last.pt"
        )
        self.MPN_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # constants
        self.t_length = 5
        self.h = 256
        self.w = 256
        self.alpha = 0.5

        self.init_variables()

    def loadMPN(self):
        # Loading the trained model
        model = torch.load(self.MPN_model_dir, map_location=self.device)
        if type(model) is dict:
            model = model["state_dict"]
        if self.device == "cuda":
            model.cuda()
        model.eval()
        return model

    def init_variables(self):
        # torch.manual_seed(2020)
        self.psnr_list = []
        self.feature_distance_list = []
        self.buffer = deque(maxlen=self.t_length)

        # torch.backends.cudnn.enabled = (
        #     True  # make sure to use cudnn for computational performance
        # )

        self.loss_func_mse = nn.MSELoss(reduction="none")
        self.MPN_model = self.loadMPN()

    def processFrame(self, img):
        with torch.no_grad():
            img = self.preprocess_image(img, resize_height=self.h, resize_width=self.w)

            self.buffer.append(img)

            # process buffer
            if len(self.buffer) == self.t_length:
                imgs = list(itertools.islice(self.buffer, 0, self.t_length))
                # concat images along the channel axis
                imgs = np.concatenate(imgs, axis=1)
                # convert the preprocessed images to a torch tensor and flash them to
                # the GPU
                imgs = torch.from_numpy(imgs)
                if self.device == "cuda":
                    imgs = imgs.cuda()

                outputs, fea_loss = self.MPN_model.forward(
                    imgs[:, : 3 * 4], weights=None, train=False
                )

                mse_imgs = self.loss_func_mse(
                    (outputs[:] + 1) / 2, (imgs[:, -3:] + 1) / 2
                )

                mse_feas = fea_loss.mean(-1)

                mse_feas = mse_feas.reshape((-1, 1, self.h, self.w))
                mse_imgs = mse_imgs.view((mse_imgs.shape[0], -1))
                mse_imgs = mse_imgs.mean(-1)
                mse_feas = mse_feas.view((mse_feas.shape[0], -1))
                mse_feas = mse_feas.mean(-1)

                for j in range(len(mse_imgs)):
                    psnr_score = self.psnr(mse_imgs[j].item())
                    fea_score = self.psnr(mse_feas[j].item())
                    self.psnr_list.append(psnr_score)
                    self.feature_distance_list.append(fea_score)

    def lastResult(self):
        template = self.calc(15, 2)
        aa = self.filter(self.anomaly_score_list(self.psnr_list[-20:]), template, 15)
        bb = self.filter(
            self.anomaly_score_list(self.feature_distance_list[-20:]), template, 15
        )
        anomaly_score_total_list = self.score_sum(aa, bb, self.alpha)
        anomaly_socre_total = np.asarray(anomaly_score_total_list)
        return anomaly_socre_total

    def previewResults(self):
        template = self.calc(15, 2)
        aa = self.filter(self.anomaly_score_list(self.psnr_list), template, 15)
        bb = self.filter(
            self.anomaly_score_list(self.feature_distance_list), template, 15
        )
        anomaly_score_total_list = self.score_sum(aa, bb, self.alpha)
        anomaly_socre_total = np.asarray(anomaly_score_total_list)
        return anomaly_socre_total

    def plotAnomaly(self, frame, score):
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        bgr = (0, 255, 0)
        cv2.putText(frame, str(score), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    def getScore_sum(self):
        return self.getScore_sum

    def getFeature_distance_list(self):
        return self.feature_distance_list

    def filter(self, data, template, radius=5):
        arr = np.array(data)
        length = arr.shape[0]
        newData = np.zeros(length)

        for j in range(radius // 2, arr.shape[0] - radius // 2):
            t = arr[j - radius // 2 : j + radius // 2 + 1]
            a = np.multiply(t, template)
            newData[j] = a.sum()
        # expand
        for i in range(radius // 2):
            newData[i] = newData[radius // 2]
        for i in range(-radius // 2, 0):
            newData[i] = newData[-radius // 2]
        # import pdb;pdb.set_trace()
        return newData

    def psnr(self, mse):
        return 10 * math.log10(1 / mse)

    def calc(self, r=5, sigma=2):
        k = np.zeros(r)
        for i in range(r):
            k[i] = (
                1
                / ((2 * math.pi) ** 0.5 * sigma)
                * math.exp(-((i - r // 2) ** 2 / 2 / (sigma ** 2)))
            )
        return k

    def score_sum(self, list1, list2, alpha):
        list_result = []
        for i in range(len(list1)):
            list_result.append((alpha * list1[i] + (1 - alpha) * list2[i]))
        return list_result

    def anomaly_score(self, psnr, max_psnr, min_psnr):
        return (psnr - min_psnr) / (max_psnr - min_psnr)

    def anomaly_score_list(self, psnr_list):
        anomaly_score_list = list()
        for i in range(len(psnr_list)):
            anomaly_score_list.append(
                self.anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list))
            )

        return anomaly_score_list

    def preprocess_image(self, image, resize_height, resize_width):
        """
        Convert image to numpy.ndarray. Notes that the color channels are BGR and the color space
        is normalized from [0, 255] to [-1, 1].
        :param filename: the full path of image
        :param resize_height: resized height
        :param resize_width: resized width
        :return: numpy.ndarray
        """
        image = cv2.resize(image, (resize_width, resize_height))
        image = image.astype(dtype=np.float32)
        image = (image / 127.5) - 1.0
        # set "channels first" ordering, and add a batch dimension
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        # return the preprocessed image
        return image
