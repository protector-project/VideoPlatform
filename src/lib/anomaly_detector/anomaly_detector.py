import itertools
import math
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.mpn import *
from models.model import create_model, load_model


class AnomalyDetector:
    def __init__(self, opt, device):
        print("Creating model...")
        self.opt = opt
        self.device = device
        self.model = create_model(opt)
        self.model = load_model(self.model, opt)
        self.model = self.model.to(device)
        self.model.eval()

        # Parameters
        self.t_length = opt.t_length
        self.h = opt.h
        self.w = opt.w
        self.alpha = opt.alpha

        self.psnr_list = []
        self.feature_distance_list = []
        self.buffer = deque(maxlen=self.t_length)
        self.loss_func_mse = nn.MSELoss(reduction="none")

        self.pred_err_buffer = []
        self.norm_err_buffer = []

    def pre_process(self, im0):
        """
        Convert image to numpy.ndarray. Notes that the color channels are BGR and the color space
        is normalized from [0, 255] to [-1, 1].
        :param filename: the full path of image
        :param resize_height: resized height
        :param resize_width: resized width
        :return: numpy.ndarray
        """
        img = cv2.resize(im0, (self.w, self.h))
        img = img.astype(dtype=np.float32)
        img = (img / 127.5) - 1.0
        # set "channels first" ordering, and add a batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        # convert the preprocessed images to a torch tensor and flash them to
        # the GPU
        img = torch.from_numpy(img).to(self.device)
        # return the preprocessed image
        return img

    def process_frame(self, im0):
        img = self.pre_process(im0.copy())

        self.buffer.append(img)

        # process buffer
        if len(self.buffer) == self.t_length:
            imgs = list(itertools.islice(self.buffer, 0, self.t_length))
            # concat images along the channel axis
            imgs = torch.cat(imgs, dim=1)

            outputs, fea_loss = self.model.forward(
                imgs[:, : 3 * 4], weights=None, train=False
            )

            mse_imgs = self.loss_func_mse((outputs[:] + 1) / 2, (imgs[:, -3:] + 1) / 2)
            self.pred_err_buffer.append(mse_imgs.squeeze(dim=0))

            mse_feas = fea_loss.mean(-1)

            mse_feas = mse_feas.reshape((-1, 1, self.w, self.h))
            self.norm_err_buffer.append(mse_feas.squeeze(dim=0))
            mse_imgs = mse_imgs.view((mse_imgs.shape[0], -1))
            mse_imgs = mse_imgs.mean(-1)
            mse_feas = mse_feas.view((mse_feas.shape[0], -1))
            mse_feas = mse_feas.mean(-1)

            for j in range(len(mse_imgs)):
                psnr_score = self.psnr(mse_imgs[j].item())
                fea_score = self.psnr(mse_feas[j].item())
                self.psnr_list.append(psnr_score)
                self.feature_distance_list.append(fea_score)
        else:
            self.pred_err_buffer.append(torch.zeros(img.shape[1:]))
            self.norm_err_buffer.append(torch.zeros(img.shape[1:]))

    def measure_anomaly_scores(self):
        template = self.calc(15, 2)
        aa = self.filter(self.anomaly_score_list(self.psnr_list), template, 15)
        bb = self.filter(
            self.anomaly_score_list(self.feature_distance_list), template, 15
        )
        anomaly_score_total_list = self.score_sum(aa, bb, self.alpha)
        anomaly_score_total = np.asarray(anomaly_score_total_list)
        return anomaly_score_total

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
