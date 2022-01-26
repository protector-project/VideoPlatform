from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent



def extant_file(fname):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(fname):
        raise argparse.ArgumentTypeError(f"{fname} does not exist")
    return fname


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # object detection
        self.parser.add_argument(
            "--detection_arch",
            default="yolo",
            help="model architecture. Currently only supports yolo",
        )

        # person detection
        self.parser.add_argument(
            "--person_detection_model",
            type=extant_file,
            default=get_project_root() / 'models/yolov5x6_mt_ft_ch.pt',
            help="path to person detection pretrained model",
        )
        self.parser.add_argument(
            "--person_detection_imgsz",
            "--person_detection_img",
            "--person_detection_img_size",
            nargs="+",
            type=int,
            default=[1280],
            help="inference size h,w",
        )

        # vehicle detection
        self.parser.add_argument(
            "--veh_detection_model",
            type=extant_file,
            default=get_project_root() / 'models/yolov5n6.pt',
            help="path to vehicle detection pretrained model",
        )
        self.parser.add_argument(
            "--veh_detection_imgsz",
            "--veh_detection_img",
            "--veh_detection_img_size",
            nargs="+",
            type=int,
            default=[1280],
            help="inference size h,w",
        )

        # anomaly detection
        self.parser.add_argument(
            "--anomaly_model",
            type=extant_file,
            default=get_project_root() / 'models/mpn_piazza_2_sett_3.pt',
            help="path to anomaly detection pretrained model",
        )
        self.parser.add_argument(
            "--anomaly_arch",
            default="mpn",
            help="model architecture. Currently only supports mpn",
        )
        self.parser.add_argument(
            "--anomaly_h", type=int, default=256, help="height of input images"
        )
        self.parser.add_argument(
            "--anomaly_w", type=int, default=256, help="width of input images"
        )
        self.parser.add_argument(
            "--t_length", type=int, default=5, help="length of the frame sequences"
        )
        self.parser.add_argument(
            "--alpha", type=float, default=0.5, help="weight for the anomality score"
        )

        # tracking
        self.parser.add_argument(
            "--tracking_model",
            type=extant_file,
            default=get_project_root() / 'models/crowdhuman_dla34.pth',
            help="path to tracking pretrained model",
        )
        self.parser.add_argument(
            "--tracking_arch",
            default="fairmot",
            help="model architecture. Currently only supports fairmot",
        )
        self.parser.add_argument(
            "--down_ratio",
            type=int,
            default=4,
            help="output stride. Currently only supports 4.",
        )
        self.parser.add_argument(
            "--input_res",
            type=int,
            default=-1,
            help="input height and width. -1 for default from "
            "dataset. Will be overriden by input_h | input_w",
        )
        self.parser.add_argument(
            "--input_h",
            type=int,
            default=-1,
            help="input height. -1 for default from dataset.",
        )
        self.parser.add_argument(
            "--input_w",
            type=int,
            default=-1,
            help="input width. -1 for default from dataset.",
        )
        self.parser.add_argument(
            "--K", type=int, default=500, help="max number of output objects."
        )
        self.parser.add_argument(
            "--conf_thres",
            type=float,
            default=0.4,
            help="confidence thresh for tracking",
        )
        self.parser.add_argument(
            "--det_thres",
            type=float,
            default=0.3,
            help="confidence thresh for detection",
        )
        self.parser.add_argument(
            "--nms_thres", type=float, default=0.4, help="iou thresh for nms"
        )
        self.parser.add_argument(
            "--track_buffer", type=int, default=30, help="tracking buffer"
        )
        self.parser.add_argument(
            "--min_box_area", type=float, default=100, help="filter out tiny boxes"
        )
        self.parser.add_argument(
            "--reid_dim", type=int, default=128, help="feature dim for reid"
        )
        self.parser.add_argument(
            "--ltrb", default=True, help="regress left, top, right, bottom of bbox"
        )

        # system
        self.parser.add_argument(
            "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="dataloader threads. 0 for single-thread.",
        )
        self.parser.add_argument(
            "--seed", type=int, default=317, help="random seed"
        )  # from CornerNet

        # IO
        self.parser.add_argument(
            "--input_video",
            type=extant_file,
            required=True,
            help="path to the input video",
        )
        self.parser.add_argument(
            "--input_name",
            type=str,
            required=True,
            help="name of the input scene",
        )
        self.parser.add_argument(
            "--output-root",
            type=str,
            default="../demos",
            help="expected output root path",
        )

        # database
        self.parser.add_argument(
            "-DH",
            "--database_host",
            type=str,
            required=False,
            help="IP address of the database",
        )
        self.parser.add_argument(
            "-DP",
            "--database_port",
            type=str,
            required=False,
            help="port number of the database",
        )
        self.parser.add_argument(
            "-DN",
            "--database_name",
            type=str,
            required=False,
            help="name of the Database",
        )

    def parse(self, args=""):
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.person_detection_imgsz *= 2 if len(opt.person_detection_imgsz) == 1 else 1  # expand
        opt.veh_detection_imgsz *= 2 if len(opt.veh_detection_imgsz) == 1 else 1  # expand

        opt.reg_offset = True

        opt.head_conv = 256

        opt.root_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        opt.save_dir = os.path.join(opt.root_dir, "exp", opt.input_name)
        print("The output will be saved to ", opt.save_dir)

        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        opt.heads = {
            "hm": opt.num_classes,
            "wh": 2 if not opt.ltrb else 4,
            "id": opt.reid_dim,
        }
        if opt.reg_offset:
            opt.heads.update({"reg": 2})
        opt.nID = dataset.nID
        opt.img_size = (1088, 608)
        # opt.img_size = (864, 480)
        # opt.img_size = (576, 320)
        print("heads", opt.heads)
        return opt

    def init(self, args=""):
        default_dataset_info = {
            "default_resolution": [608, 1088],
            "num_classes": 1,
            "mean": [0.408, 0.447, 0.470],
            "std": [0.289, 0.274, 0.278],
            "dataset": "jde",
            "nID": 14455,
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        opt = self.parse(args)
        dataset = Struct(default_dataset_info)
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt
