from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch

from lib.models.yolo import get_yolo
from lib.models.mpn import get_mpn
from lib.models.networks.pose_dla_dcn import get_fairmot
from lib.models.ynet import get_ynet
from lib.models.strong_sort.strong_sort import get_strong_sort

_model_factory = {
    "yolo": get_yolo,
    "mpn": get_mpn,
    "fairmot": get_fairmot,
    "ynet": get_ynet,
    "strong_sort": get_strong_sort
}


def create_model(opt):
    get_model = _model_factory[opt.arch]
    model = get_model(opt)
    return model


def load_model(model, opt, optimizer=None, resume=False, lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
    print("loaded {}, epoch {}".format(opt.model_path, checkpoint.get("epoch", -1)))
    if opt.arch == "yolo":
        state_dict_ = checkpoint.get("model").float().state_dict()
    elif opt.arch == "mpn" or opt.arch == "fairmot":
        state_dict_ = checkpoint.get("state_dict")
    elif opt.arch == "ynet":
        state_dict_ = checkpoint
    else:
        raise ValueError("Invalid model architecture, expected one of {yolo, fairmot, mpn, ynet}")
    state_dict = {}

    # convert data_parallel to model
    for k in state_dict_:
        if k.startswith("module") and not k.startswith("module_list"):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = (
        "If you see this, your model does not fully load the "
        + "pre-trained weight. Please make sure "
        + "you have correctly specified --arch xxx "
        + "or set the correct --num_classes for your own dataset."
    )
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(
                    "Skip loading parameter {}, required shape{}, "
                    "loaded shape{}. {}".format(
                        k, model_state_dict[k].shape, state_dict[k].shape, msg
                    )
                )
                state_dict[k] = model_state_dict[k]
        else:
            print("Drop parameter {}.".format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print("No param {}.".format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = start_lr
            print("Resumed optimizer with start lr", start_lr)
        else:
            print("No optimizer parameters in checkpoint.")
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {"epoch": epoch, "state_dict": state_dict}
    if not (optimizer is None):
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, path)
