from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import _init_paths

import os.path
import cv2
import torch
import influxdb
import pandas as pd

from tqdm import tqdm
from lib.anomaly_detector.traj_anomaly_detector import TrajAnomalyDetector

from lib.object_detector.object_detector import ObjectDetector
from lib.anomaly_detector.anomaly_detector import AnomalyDetector
from lib.influx.influx_client import InfluxClient
from lib.datasets.dataset import LoadImages
from lib.opts import get_project_root, opts
from lib.tracker.tracker import Tracker
from lib.utils.preprocessing import load_and_window_MT

from lib.utils.torch_utils import select_device
from lib.utils.visualization import plot_anomaly, plot_boxes, plot_tracking, plot_trajectories


VEHICLES_LABELS = ["bicycle", "car", "motorcycle", "bus", "truck"]


@torch.no_grad()
def main(opt):
	# Load models
	device = select_device(opt.device)
	anomaly_detector = AnomalyDetector(opt.img_anomaly_detection, device)
	person_detector = ObjectDetector(
		opt.person_detection, device
	)
	veh_detector = ObjectDetector(
		opt.veh_detection, device
	)
	# tracker = Tracker(opt.tracking, device)
	# traj_anomaly_detector = TrajAnomalyDetector(opt.traj_anomaly_detection, device)

	veh_classes = [veh_detector.model.names.index(veh) for veh in VEHICLES_LABELS]

	if opt.use_database:
		influx = InfluxClient(
			host=opt.database_host,
			port=int(opt.database_port),
			database=opt.database_name,
		)
		influx.createConnection()

	count = -20

	dataset = LoadImages(opt.input_video)

	frame_id = 0
	trajectories = []
	with tqdm(total=len(dataset)) as pbar:
		# Run inference
		for path, im0s, vid_cap, s in dataset:
			pbar.set_description("Processing %s" % path)

			# Inference

			################################################ video object detection ################################################
			person_results = person_detector.process_frame(im0s)
			veh_results = veh_detector.process_frame(im0s, veh_classes)
			img = plot_boxes(person_results + veh_results, im0s)

			################################################ video anomaly detection (image) ################################################
			anomaly_detector.process_frame(im0s)

			################################################ video object tracking ################################################
			# online_tlwhs, online_ids = tracker.process_frame(im0s)
			# trajectories.extend(
			# 	[
			# 		(frame_id, track_id, *tlwh)
			# 		for track_id, tlwh in zip(online_ids, online_tlwhs)
			# 	]
			# )
			# img = plot_tracking(im0s, online_tlwhs, online_ids, frame_id=frame_id)

			################################################ visualization (detection/tracking) ################################################
			# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
			# cv2.setWindowProperty(
			# 	"frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
			# )
			# cv2.imshow("frame", img)
			# if cv2.waitKey(1) & 0xFF == ord("q"):
			# 	cv2.destroyAllWindows()
			# 	vid_cap.release()
			# 	break

			# if opt.use_database:
			#     frame_time = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
			#     if person_results != []:
			#         count_label = person_detector.count_label(person_results, "0")
			#         influx.insertHumans(opt.input_name, count_label, frame_time)
			#     if veh_results != []:
			#         influx.insertObjects(
			#             opt.input_name, opt.input_video, veh_results, frame_time
			#         )

			frame_id += 1
			pbar.update(1)

	################################################ video anomaly detection (image) ################################################
	anomaly_scores = anomaly_detector.measure_anomaly_scores()

	################################################ video anomaly detection (trajectory) ################################################
	# scene_df = load_and_window_MT(
	# 	trajectories=trajectories,
	# 	scene=opt.input_name,
	# 	step=5,
	# 	window_size=20,
	# 	stride=20,
	# )
	# obs_len = opt.traj_anomaly_detection.obs_len
	# pred_len = opt.traj_anomaly_detection.pred_len

	dataset = LoadImages(opt.input_video)
	frame_id = 0
	for i, (path, im0s, vid_cap, s) in enumerate(dataset):

		################################################ video anomaly detection (image) ################################################
		score_idx = i - (anomaly_detector.t_length - 1)
		anomaly_score = anomaly_scores[score_idx] if score_idx >= 0 else -1
		img = plot_anomaly(im0s, anomaly_score, frame_id=frame_id)

		################################################ visualization (anomaly) ################################################
		cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
		cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		cv2.imshow('frame', img)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			cv2.destroyAllWindows()
			vid_cap.release()
			break

		################################################ video anomaly detection (trajectory) ################################################
		# if frame_id in scene_df["frame"].values:
		# 	trajectories = []
		# 	track_ids = scene_df.loc[(scene_df["frame"] == frame_id)]["trackId"].values
		# 	for track_id in track_ids:
		# 		obs_df = scene_df.loc[
		# 			(scene_df["frame"] <= frame_id) & (scene_df["trackId"] == track_id)
		# 		]
		# 		pred_df = scene_df.loc[
		# 			(scene_df["frame"] > frame_id) & (scene_df["trackId"] == track_id)
		# 		]
		# 		if len(obs_df) >= obs_len and len(pred_df) >= pred_len:
		# 			obs_traj = (
		# 				obs_df[["x", "y"]]
		# 				.to_numpy()
		# 				.astype("float32")[-obs_len :]
		# 				.reshape(obs_len, 2)
		# 			)
		# 			pred_traj = (
		# 				pred_df[["x", "y"]]
		# 				.to_numpy()
		# 				.astype("float32")[: pred_len]
		# 				.reshape(pred_len, 2)
		# 			)
		# 			trajectories.append(np.concatenate([obs_traj, pred_traj]))
		# 	if trajectories:
		# 		(
		# 			future_samples_list,
		# 			waypoint_samples_list,
		# 		) = traj_anomaly_detector.process_frame(
		# 			im0s, torch.Tensor(trajectories), opt.input_name
		# 		)
		# 		val_ADE, val_FDE = traj_anomaly_detector.measure_anomaly_score(
		# 			torch.Tensor(trajectories),
		# 			future_samples_list,
		# 			waypoint_samples_list,
		# 		)
		# 		print(f"val_ADE: {val_ADE} - val_FDE: {val_FDE}")
		# 		traj_anomaly_detector.plot_results(im0s, torch.Tensor(trajectories), future_samples_list)

		# if opt.use_database:
		#     frame_time = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
		#     influx.insertAnomaly(opt.input_name, float(anomaly_score), opt.input_video, frame_time)

		frame_id += 1


if __name__ == "__main__":
	opt = opts().init()
	main(opt)
