import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

def select_device(device=""):
    device = (
        str(device).strip().lower().replace("cuda:", "")
    )  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    if cpu:
        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        assert (
            torch.cuda.is_available()
        ), f"CUDA unavailable, invalid device {device} requested"  # check availability

    cuda = not cpu and torch.cuda.is_available()
    return torch.device("cuda:0" if cuda else "cpu")


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # Scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True