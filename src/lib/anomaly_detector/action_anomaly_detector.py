from email.mime import image
import itertools
import math
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from PIL import Image
from models.clip import *
from models.model import create_model, load_model
from utils.augmentation import get_inference_augmentation
from utils.text_prompt import text_prompt
from pytorch_grad_cam import GradCAM
from utils.tools import convert_models_to_fp32


class SimilarityToTextFeaturesTarget:
    def __init__(self, text_features):
        self.text_features = text_features

    def __call__(self, image_features):
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # self.text_features = self.text_features / self.text_features.norm(
        #     dim=-1, keepdim=True
        # )

        # cosine similarity
        similarity = 100.0 * image_features @ self.text_features.T
        return similarity


def reshape_transform(tensor, height=14, width=14):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class ActionAnomalyDetector:
    def __init__(self, opt, device):
        print("Creating model...")
        self.opt = opt
        self.device = device
        self.action_clip = create_model(opt)
        # self.model, self.fusion_model = self.action_clip.model, self.action_clip.fusion_model
        # # self.model = load_model(self.model, opt)
        # convert_models_to_fp32(self.model)
        # convert_models_to_fp32(self.fusion_model)
        convert_models_to_fp32(self.action_clip)
        # self.model.eval()
        # self.fusion_model.eval()
        # self.model = self.model.to(device)
        # self.fusion_model = self.fusion_model.to(device)
        self.action_clip.eval()
        self.action_clip = self.action_clip.to(device)

        # Parameters
        self.dataset = opt.data.dataset
        self.seg_length = opt.data.seg_length

        self.transform = get_inference_augmentation(opt)

        classes_all = pd.read_csv(opt.data.label_list)
        self.normal_idx = classes_all.loc[classes_all['name'] == 'normal', 'id'].iloc[0]
        self.classes_all = classes_all.values.tolist()
        self.classes, self.num_text_aug, self.text_dict = text_prompt(self.dataset)
        self.text_inputs = self.classes.to(device)  # [36, 77]
        self.text_features = self.action_clip.model.encode_text(self.text_inputs)  # [36, 512]
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)  # [36, 512]

        self.buffer = list()
        self.y_values = list()
        self.y_indices = list()
        self.anomaly_scores = list()
        self.psnr_list = list()

        self.gradcam_images = list()
        # self.action_clip.eval()
        # self.action_clip = self.action_clip.to(device)

    def pre_process(self, im0):
        # Convert
        im = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        return im

    def process_frame(self, im0):
        im = self.pre_process(im0.copy())

        self.buffer.append(im)

        # process buffer
        if len(self.buffer) == self.seg_length:
            process_data = self.transform(self.buffer)  # [48, 224, 224]
            process_data = torch.unsqueeze(process_data, 0)  # [1, 48, 224, 224]
            image = process_data.view(
                (-1, self.seg_length, 3) + process_data.size()[-2:]  # [1, 16, 3, 224, 224]
            )
            b, t, c, h, w = image.size()
            image_input = image.to(self.device).view(-1, c, h, w)  # [16, 3, 224, 224]
            image_features = self.action_clip.model.encode_image(image_input).view(b, t, -1)  # [1, 16, 512]
            image_features = self.action_clip.fusion_model(image_features)  # [1, 512]
            image_features /= image_features.norm(dim=-1, keepdim=True)  # [1, 512]
            similarity = 100.0 * image_features @ self.text_features.T
            similarity = similarity.view(b, -1, self.num_text_aug).mean(dim=-1)  # [1, 4]
            similarity = similarity.softmax(dim=1)  # [1, 4]
            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_4, indices_4 = similarity.topk(4, dim=-1)
            values_4_list = torch.squeeze(values_4, dim=0).tolist()
            indices_4_list = torch.squeeze(indices_4, dim=0).tolist()
            self.y_values.extend([values_4_list] * len(self.buffer))
            self.y_indices.extend([indices_4_list] * len(self.buffer))
            anomaly_score = 1 - values_4_list[indices_4_list.index(self.normal_idx)]
            self.anomaly_scores.extend([anomaly_score] * len(self.buffer))
            # self.psnr_list.extend([self.psnr(anomaly_score)] * len(self.buffer))

            ### GradCAM
            text_features = self.text_features.view(
                -1, self.num_text_aug, self.text_features.shape[1]  # [4, 9, 512]
            )
            text_features = text_features.mean(dim=1, keepdim=False)  # [4, 512]
            most_sim_idx = indices_1.item()
            text_features = torch.unsqueeze(self.text_features[most_sim_idx], dim=0)  # [1, 512]
            text_targets = [SimilarityToTextFeaturesTarget(text_features)] 
            use_cuda = torch.cuda.is_available()
            target_layers = [self.action_clip.model.visual.transformer.resblocks[-1].ln_1]
            # target_layers = [self.action_clip.fusion_model.transformer.resblocks[-1].ln_1]
            cam = GradCAM(
                model=self.action_clip,
                target_layers=target_layers,
                use_cuda=use_cuda,
                reshape_transform=reshape_transform,
            )
            grayscale_cam = cam(
                input_tensor=image,
                targets=text_targets,
                eigen_smooth=False,
                aug_smooth=False,
            )
            # grayscale_cam = grayscale_cam[0, :]
            # mu, sigma = 0, 0.1 # mean and standard deviation
            # grayscale_cam = np.random.normal(mu, sigma, size=(224, 224))
            # self.gradcam_images.extend([grayscale_cam] * len(self.buffer))
            self.gradcam_images.extend([grayscale_cam[i] for i in range(grayscale_cam.shape[0])])
            ###

            self.buffer = list()

    # def measure_anomaly_scores(self):
    #     template = self.calc(15, 2)
    #     anomaly_score_total_list = self.filter(self.anomaly_score_list(self.psnr_list), template, 15)
    #     anomaly_score_total = np.asarray(anomaly_score_total_list)
    #     return anomaly_score_total

    # def filter(self, data, template, radius=5):
    #     arr = np.array(data)
    #     length = arr.shape[0]
    #     newData = np.zeros(length)

    #     for j in range(radius // 2, arr.shape[0] - radius // 2):
    #         t = arr[j - radius // 2 : j + radius // 2 + 1]
    #         a = np.multiply(t, template)
    #         newData[j] = a.sum()
    #     expand
    #     for i in range(radius // 2):
    #         newData[i] = newData[radius // 2]
    #     for i in range(-radius // 2, 0):
    #         newData[i] = newData[-radius // 2]
    #     import pdb;pdb.set_trace()
    #     return newData

    # def psnr(self, mse):
    #     return 10 * math.log10(1 / mse)

    # def calc(self, r=5, sigma=2):
    #     k = np.zeros(r)
    #     for i in range(r):
    #         k[i] = (
    #             1
    #             / ((2 * math.pi) ** 0.5 * sigma)
    #             * math.exp(-((i - r // 2) ** 2 / 2 / (sigma ** 2)))
    #         )
    #     return k

    # def score_sum(self, list1, list2, alpha):
    #     list_result = []
    #     for i in range(len(list1)):
    #         list_result.append((alpha * list1[i] + (1 - alpha) * list2[i]))
    #     return list_result

    # def anomaly_score(self, psnr, max_psnr, min_psnr):
    #     return (psnr - min_psnr) / (max_psnr - min_psnr)

    # def anomaly_score_list(self, psnr_list):
    #     anomaly_score_list = list()
    #     for i in range(len(psnr_list)):
    #         anomaly_score_list.append(
    #             self.anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list))
    #         )

    #     return anomaly_score_list