# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu
# the one called
import torch
from lib.models.clip import clip
from collections import OrderedDict


def text_prompt(dataset):
    text_dict = {}

    text_aug = [
        f"a photo of a person {{}}",
        f"a person {{}} on the street",
        f"people {{}}",
        f"human action of {{}}",
        f"a group of people {{}}",
        f"a photo of someone {{}}",
        f"people {{}} on the street",
        f"Look, the human is {{}}",
        f"{{}} on the street",
    ]
    text_aug3 = [f"{{}}"] * 9

    num_text_aug = len(text_aug)

    if dataset == "mt":
        verbs = [[0, "dealing drugs"], [1, "fighting"], [3, "running"]]

        # nouns = [[0, "drug dealing"]]

        other = [[2, "normal"]]

        for i in range(len(verbs)):
            text_dict[verbs[i][0]] = torch.cat(
                [clip.tokenize(txt.format(verbs[i][1])) for txt in text_aug]
            )

        for i in range(len(other)):
            text_dict[other[i][0]] = torch.cat(
                [clip.tokenize(txt.format(other[i][1])) for txt in text_aug3]
            )

    else:
        raise ValueError("Not supported dataset")

    sorted_text_dict = dict(sorted(text_dict.items()))

    classes = torch.cat([v for k, v in sorted_text_dict.items()])

    return classes, num_text_aug, sorted_text_dict
