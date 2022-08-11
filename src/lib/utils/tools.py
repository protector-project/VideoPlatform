# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import numpy
import cv2
import torch
from pathlib import Path
import os 

def gen_label(labels):
    num = len(labels)
    gt = numpy.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    
    ## genrates a batch_size*batch_size tensor with 1 in the i-th row (or col) where labels id are equal
    
    # tensor([17, 17,  0, 17, 17, 17, 17, 17])
    #        [[1. 1. 0. 1. 1. 1. 1. 1.]
    #         [1. 1. 0. 1. 1. 1. 1. 1.]
    #         [0. 0. 1. 0. 0. 0. 0. 0.]
    #         [1. 1. 0. 1. 1. 1. 1. 1.]
    #         [1. 1. 0. 1. 1. 1. 1. 1.]
    #         [1. 1. 0. 1. 1. 1. 1. 1.]
    #         [1. 1. 0. 1. 1. 1. 1. 1.]
    #         [1. 1. 0. 1. 1. 1. 1. 1.]]

    return gt

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def convert_models_to_fp16(model):
    print(model)
    for p in model.parameters():
        p.data = p.data.half()
        p.grad.data = p.grad.data.half()


def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2
    

def get_color(idx):
	idx = idx * 3
	color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

	return color


def visualize_vid_with_top1class(vid_file, pred_indices, pred_values, classes, output_dir="/usr/src/app/runs/mt/"):
    vid_cap = cv2.VideoCapture(vid_file)
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    video_name = vid_file.split("/")[-1]

    output_path = os.path.join(output_dir, video_name)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    f_idx = 0
    success, frame = vid_cap.read()

    while success and f_idx < len(pred_indices):
        score = pred_values[f_idx].item()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness=2

        text_scale = max(font_scale, frame.shape[1] / 1600.0)
        top_left_corner_of_text = (0, int(30 * text_scale))
        bottom_left_corner_of_text = (0, 200)
        
        cls, label = classes[pred_indices[f_idx]]
        text = label
        color = get_color(int(cls))

        cv2.putText(
                frame,
                f"{text}: {score:.6f}",
                top_left_corner_of_text,
                font,
                text_scale,
                color,
                thickness=thickness,
            )

        out.write(frame)
    
        success, frame = vid_cap.read()
        f_idx += 1
    
    cv2.destroyAllWindows()
    vid_cap.release()


def visualize_vid_with_topkclasses(vid_file, pred_indices, pred_values, classes, output_dir="/usr/src/app/runs/mt/"):
    vid_cap = cv2.VideoCapture(vid_file)
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    video_name = vid_file.split("/")[-1]

    output_path = os.path.join(output_dir, video_name)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    f_idx = 0
    success, frame = vid_cap.read()

    while success and f_idx < len(pred_indices):
        position = (0, 30)
        pred_values_list = pred_values[f_idx].view(-1).tolist()
        pred_indices_list = pred_indices[f_idx].view(-1).tolist()
        text = [f"{classes[idx][1]}: {score:.6f}" for idx, score in zip(pred_indices_list, pred_values_list)]
        text = "\n".join(text)
        font_scale = 0.8
        thickness=2
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_type = cv2.LINE_AA
        
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        line_height = text_size[1] + 5
        x, y0 = position
        
        for i, line in enumerate(text.split("\n")):
            y = y0 + i * line_height
            color = get_color(int(pred_indices_list[i]))
            cv2.putText(
                frame,
                line,
                (x, y),
                font,
                font_scale,
                color,
                thickness,
                line_type
            )

        out.write(frame)
    
        success, frame = vid_cap.read()
        f_idx += 1
    
    cv2.destroyAllWindows()
    vid_cap.release()