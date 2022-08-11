from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import yaml
from pathlib import Path
from dotmap import DotMap


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def extant_file(fname):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(fname):
        raise argparse.ArgumentTypeError(f"{fname} does not exist")
    return fname


class Struct:
    def __init__(self, entries):
        for k, v in entries.items():
            self.__setattr__(k, v)


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
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
            "--output_root",
            type=str,
            default=get_project_root() / "demos",
            help="expected output root path",
        )

        # config
        self.parser.add_argument(
            "--config_file",
            type=extant_file,
            required=True,
            help="path to config file",
        )  

    def parse(self, args=""):
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        # Load config file and add to command-line-arguments
        # with open(opt.config_file) as file:
        #     params = yaml.load(file, Loader=yaml.FullLoader)
        #     delattr(opt, 'config_file')
        #     opt_dict = opt.__dict__
        #     for key, value in params.items():
        #         if isinstance(value, dict):
        #             opt_dict[key] = Struct(value)
        #         elif isinstance(value, list):
        #             for v in value:
        #                 opt_dict[key].extend(v)
        #         else:
        #             opt_dict[key] = value

        # Load config file and add to command-line-arguments
        with open(opt.config_file) as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
            delattr(opt, 'config_file')
        opt_dict = opt.__dict__
        opt = DotMap(params)
        for key, value in opt_dict.items():
            opt[key] = value

        opt.object_detection.imgsz *= (
            2 if len(opt.object_detection.imgsz) == 1 else 1
        )  # expand

        # opt.root_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        # opt.save_dir = os.path.join(opt.root_dir, "exp", opt.input_name)
        Path(opt.output_root).mkdir(parents=True, exist_ok=True)
        print("The output will be saved to ", opt.output_root)

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

        opt = self.parse(args)
        dataset = Struct(default_dataset_info)
        opt.dataset = dataset.dataset
        # opt.tracking = self.update_dataset_info_and_set_heads(opt.tracking, dataset)
        return opt
