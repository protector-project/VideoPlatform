import itertools
import math
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from lib.opts import get_project_root
from lib.utils.image import (
    create_dist_mat,
    get_patch,
    pad,
    preprocess_image_for_segmentation,
    resize,
    sampling,
)
from lib.utils.kmeans import kmeans
from lib.utils.visualization import plot_trajectories

from models.ynet import *
from models.model import create_model, load_model

import skimage.io


def torch_multivariate_gaussian_heatmap(
    coordinates, H, W, dist, sigma_factor, ratio, device, rot=False
):
    """
    Create Gaussian Kernel for CWS
    """
    ax = torch.linspace(0, H, H, device=device) - coordinates[1]
    ay = torch.linspace(0, W, W, device=device) - coordinates[0]
    xx, yy = torch.meshgrid([ax, ay])
    meshgrid = torch.stack([yy, xx], dim=-1)
    radians = torch.atan2(dist[0], dist[1])

    c, s = torch.cos(radians), torch.sin(radians)
    R = torch.Tensor([[c, s], [-s, c]]).to(device)
    if rot:
        R = torch.matmul(torch.Tensor([[0, -1], [1, 0]]).to(device), R)
    dist_norm = (
        dist.square().sum(-1).sqrt() + 5
    )  # some small padding to avoid division by zero

    conv = torch.Tensor(
        [[dist_norm / sigma_factor / ratio, 0], [0, dist_norm / sigma_factor]]
    ).to(device)
    conv = torch.square(conv)
    T = torch.matmul(R, conv)
    T = torch.matmul(T, R.T)

    kernel = (torch.matmul(meshgrid, torch.inverse(T)) * meshgrid).sum(-1)
    kernel = torch.exp(-0.5 * kernel)
    return kernel / kernel.sum()


class TrajAnomalyDetector:
    def __init__(self, opt, device):
        print("Creating model...")
        self.opt = opt
        self.device = device
        self.model = create_model(opt.traj_anomaly_arch)
        self.model = load_model(
            self.model, opt.traj_anomaly_model, opt.traj_anomaly_arch
        )
        self.model = self.model.to(device)
        self.model.eval()

        # Parameters
        self.obs_len = opt.obs_len  # in timesteps
        self.pred_len = opt.pred_len  # in timesteps
        self.total_len = self.pred_len + self.obs_len
        self.waypoints = opt.waypoints
        self.temperature = opt.temperature
        self.num_goals = opt.num_goals  # K_e
        self.num_traj = opt.num_traj  # K_a
        self.resize = opt.resize
        self.seg_mask = False
        self.division_factor = 2 ** len([32, 32, 64, 64, 64])
        self.use_TTST = opt.use_TTST
        self.rel_thresh = opt.rel_thresh
        self.use_CWS = opt.use_CWS
        self.CWS_params = None

        # Create template
        size = int(4200 * opt.resize)
        input_template = create_dist_mat(size=size)
        self.input_template = torch.Tensor(input_template).to(device)

    def pre_process(self, im0):
        im = {self.opt.input_name: im0}
        # Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
        resize(im, factor=self.resize, seg_mask=self.seg_mask)
        pad(
            im, division_factor=self.division_factor
        )  # make sure that image shape is divisible by 32, for UNet segmentation
        preprocess_image_for_segmentation(im, seg_mask=self.seg_mask)
        return im

    def process_frame(self, im0, trajectories):
        im = self.pre_process(im0.copy())
        scene_image = im[self.opt.input_name].to(self.device).unsqueeze(0)
        scene_image = self.model.segmentation(scene_image)
        future_samples_list = []
        waypoint_samples_list = []

        counter = 0
        for i in range(len(trajectories)):
            # Create Heatmaps for past and ground-truth future trajectories
            _, _, H, W = scene_image.shape
            observed = (
                trajectories[i : i + 1, : self.obs_len, :].reshape(-1, 2).cpu().numpy()
            )
            observed_map = get_patch(self.input_template, observed, H, W)
            observed_map = torch.stack(observed_map).reshape([-1, self.obs_len, H, W])

            # semantic_image = scene_image.expand(observed_map.shape[0], -1, -1, -1)

            # Forward pass
            # Calculate features
            # feature_input = torch.cat([semantic_image, observed_map], dim=1)
            feature_input = observed_map
            features = self.model.pred_features(feature_input)

            # Predict goal and waypoint probability distributions
            pred_waypoint_map = self.model.pred_goal(features)
            pred_waypoint_map = pred_waypoint_map[:, self.waypoints]

            pred_waypoint_map_sigmoid = pred_waypoint_map / self.temperature
            pred_waypoint_map_sigmoid = self.model.sigmoid(pred_waypoint_map_sigmoid)

            ################################################ TTST ##################################################
            if self.use_TTST:
                # TTST Begin
                # sample a large amount of goals to be clustered
                goal_samples = sampling(
                    pred_waypoint_map_sigmoid[:, -1:],
                    num_samples=10000,
                    replacement=True,
                    rel_threshold=self.rel_thresh,
                )
                goal_samples = goal_samples.permute(2, 0, 1, 3)

                num_clusters = self.num_goals - 1
                goal_samples_softargmax = self.model.softargmax(
                    pred_waypoint_map[:, -1:]
                )  # first sample is softargmax sample

                # Iterate through all person/batch_num, as this k-Means implementation doesn't support batched clustering
                goal_samples_list = []
                for person in range(goal_samples.shape[1]):
                    goal_sample = goal_samples[:, person, 0]

                    # Actual k-means clustering, Outputs:
                    # cluster_ids_x -  Information to which cluster_idx each point belongs to
                    # cluster_centers - list of centroids, which are our new goal samples
                    cluster_ids_x, cluster_centers = kmeans(
                        X=goal_sample,
                        num_clusters=num_clusters,
                        distance="euclidean",
                        device=self.device,
                        tqdm_flag=False,
                        tol=0.001,
                        iter_limit=1000,
                    )
                    goal_samples_list.append(cluster_centers)

                goal_samples = (
                    torch.stack(goal_samples_list).permute(1, 0, 2).unsqueeze(2)
                )
                goal_samples = torch.cat(
                    [goal_samples_softargmax.unsqueeze(0), goal_samples], dim=0
                )
                # TTST End

            # Not using TTST
            else:
                goal_samples = sampling(
                    pred_waypoint_map_sigmoid[:, -1:], num_samples=self.num_goals
                )
                goal_samples = goal_samples.permute(2, 0, 1, 3)

            # Predict waypoints:
            # in case len(waypoints) == 1, so only goal is needed (goal counts as one waypoint in this implementation)
            if len(self.waypoints) == 1:
                waypoint_samples = goal_samples

            ################################################ CWS ###################################################
            # CWS Begin
            if self.use_CWS and len(self.waypoints) > 1:
                sigma_factor = self.CWS_params["sigma_factor"]
                ratio = self.CWS_params["ratio"]
                rot = self.CWS_params["rot"]

                goal_samples = goal_samples.repeat(
                    self.num_traj, 1, 1, 1
                )  # repeat K_a times
                last_observed = trajectories[i : i + 1, self.obs_len - 1].to(
                    self.device
                )  # [N, 2]
                waypoint_samples_list = (
                    []
                )  # in the end this should be a list of [K, N, # waypoints, 2] waypoint coordinates
                for g_num, waypoint_samples in enumerate(goal_samples.squeeze(2)):
                    waypoint_list = []  # for each K sample have a separate list
                    waypoint_list.append(waypoint_samples)

                    for waypoint_num in reversed(range(len(self.waypoints) - 1)):
                        distance = last_observed - waypoint_samples
                        gaussian_heatmaps = []
                        traj_idx = (
                            g_num // self.num_goals
                        )  # idx of trajectory for the same goal
                        for dist, coordinate in zip(
                            distance, waypoint_samples
                        ):  # for each person
                            length_ratio = 1 / (waypoint_num + 2)
                            gauss_mean = coordinate + (
                                dist * length_ratio
                            )  # Get the intermediate point's location using CV model
                            sigma_factor_ = sigma_factor - traj_idx
                            gaussian_heatmaps.append(
                                torch_multivariate_gaussian_heatmap(
                                    gauss_mean,
                                    H,
                                    W,
                                    dist,
                                    sigma_factor_,
                                    ratio,
                                    self.device,
                                    rot,
                                )
                            )
                        gaussian_heatmaps = torch.stack(gaussian_heatmaps)  # [N, H, W]

                        waypoint_map_before = pred_waypoint_map_sigmoid[:, waypoint_num]
                        waypoint_map = waypoint_map_before * gaussian_heatmaps
                        # normalize waypoint map
                        waypoint_map = (
                            waypoint_map.flatten(1)
                            / waypoint_map.flatten(1).sum(-1, keepdim=True)
                        ).view_as(waypoint_map)

                        # For first traj samples use softargmax
                        if g_num // self.num_goals == 0:
                            # Softargmax
                            waypoint_samples = self.model.softargmax_on_softmax_map(
                                waypoint_map.unsqueeze(0)
                            )
                            waypoint_samples = waypoint_samples.squeeze(0)
                        else:
                            waypoint_samples = sampling(
                                waypoint_map.unsqueeze(1),
                                num_samples=1,
                                rel_threshold=0.05,
                            )
                            waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
                            waypoint_samples = waypoint_samples.squeeze(2).squeeze(0)
                        waypoint_list.append(waypoint_samples)

                    waypoint_list = waypoint_list[::-1]
                    waypoint_list = torch.stack(waypoint_list).permute(
                        1, 0, 2
                    )  # permute back to [N, # waypoints, 2]
                    waypoint_samples_list.append(waypoint_list)
                waypoint_samples = torch.stack(waypoint_samples_list)

                # CWS End

            # If not using CWS, and we still need to sample waypoints (i.e., not only goal is needed)
            elif not self.use_CWS and len(self.waypoints) > 1:
                waypoint_samples = sampling(
                    pred_waypoint_map_sigmoid[:, :-1],
                    num_samples=self.num_goals * self.num_traj,
                )
                waypoint_samples = waypoint_samples.permute(2, 0, 1, 3)
                goal_samples = goal_samples.repeat(
                    self.num_traj, 1, 1, 1
                )  # repeat K_a times
                waypoint_samples = torch.cat([waypoint_samples, goal_samples], dim=2)

            # Interpolate trajectories given goal and waypoints
            future_samples = []
            for waypoint in waypoint_samples:
                waypoint_map = get_patch(
                    self.input_template, waypoint.reshape(-1, 2).cpu().numpy(), H, W
                )
                waypoint_map = torch.stack(waypoint_map).reshape(
                    [-1, len(self.waypoints), H, W]
                )

                waypoint_maps_downsampled = [
                    nn.AvgPool2d(kernel_size=2 ** i, stride=2 ** i)(waypoint_map)
                    for i in range(1, len(features))
                ]
                waypoint_maps_downsampled = [waypoint_map] + waypoint_maps_downsampled

                traj_input = [
                    torch.cat([feature, goal], dim=1)
                    for feature, goal in zip(features, waypoint_maps_downsampled)
                ]

                pred_traj_map = self.model.pred_traj(traj_input)
                pred_traj = self.model.softargmax(pred_traj_map)
                future_samples.append(pred_traj)

            future_samples = torch.stack(future_samples)

            future_samples_list.append(future_samples)
            waypoint_samples_list.append(waypoint_samples)

        return future_samples_list, waypoint_samples_list

    def measure_anomaly_score(
        self, trajectories, future_samples_list, waypoint_samples_list
    ):
        val_ADE = []
        val_FDE = []

        for i in range(len(trajectories)):
            gt_future = trajectories[i : i + 1, self.obs_len :].to(self.device)
            gt_goal = gt_future[:, -1:]

            future_samples = future_samples_list[i]
            waypoint_samples = waypoint_samples_list[i]

            val_FDE.append(
                (
                    (((gt_goal - waypoint_samples[:, :, -1:]) / self.resize) ** 2).sum(
                        dim=3
                    )
                    ** 0.5
                ).min(dim=0)[0]
            )
            val_ADE.append(
                ((((gt_future - future_samples) / self.resize) ** 2).sum(dim=3) ** 0.5)
                .mean(dim=2)
                .min(dim=0)[0]
            )

        val_ADE = torch.cat(val_ADE).mean()
        val_FDE = torch.cat(val_FDE).mean()

        return val_ADE.item(), val_FDE.item()

    def plot_results(self, im0, trajectories, future_samples_list):
        for i in range(len(trajectories)):
            observed = (
                trajectories[i : i + 1, : self.obs_len, :].reshape(-1, 2).cpu().numpy()
            )
            gt_future = trajectories[i : i + 1, self.obs_len :].to(self.device)
            future_samples = future_samples_list[i]
            plot_trajectories(
                gt_future,
                future_samples,
                observed,
                im0,
                self.resize,
                with_bg=True,
            )
