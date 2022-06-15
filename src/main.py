from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import enum
from pathlib import Path

import numpy as np

import _init_paths

import os.path
import cv2
import torch
import influxdb
import pandas as pd
import datetime

from tqdm import tqdm
from lib.anomaly_detector.traj_anomaly_detector import TrajAnomalyDetector

from lib.object_detector.object_detector import ObjectDetector
from lib.anomaly_detector.anomaly_detector import AnomalyDetector
from lib.influx.influx_client import InfluxClient
from lib.influx.json_influx import InfluxJson
from lib.influx.video_output import VideoOutput
from lib.datasets.dataset import LoadImages
from lib.opts import get_project_root, opts
from lib.tracker.tracker import Tracker
from lib.utils.preprocessing import load_and_window_MT

from lib.utils.torch_utils import select_device
from lib.utils.visualization import plot_anomaly, plot_boxes, plot_pred_err, plot_norm_err, plot_tracking, plot_trajectories


VEHICLES_LABELS = ["bicycle", "car", "motorcycle", "bus", "truck"]
OBJECTS_LABELS = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']


@torch.no_grad()
def main(opt):
	# Load models
	device = select_device(opt.device)
	if opt.img_anomaly_detection.enabled:
		anomaly_detector = AnomalyDetector(opt.img_anomaly_detection, device)
	if opt.person_detection.enabled:
		person_detector = ObjectDetector(
			opt.person_detection, device
		)
		person_index = person_detector.model.names.index("person")
	if opt.veh_detection.enabled:
		veh_detector = ObjectDetector(
			opt.veh_detection, device
		)
		veh_classes = [veh_detector.model.names.index(veh) for veh in VEHICLES_LABELS]
	if opt.tracking.enabled:
		tracker = Tracker(opt.tracking, device)
	if opt.traj_anomaly_detection.enabled:
		traj_anomaly_detector = TrajAnomalyDetector(opt.traj_anomaly_detection, device)

	if opt.use_database:
		influx = InfluxClient(
			host=opt.database_host,
			port=int(opt.database_port),
			database=opt.database_name,
		)
		influx.createConnection()
  
	if opt.produce_files.enable:
		video_output = VideoOutput(opt.input_video)
		json_output = InfluxJson(opt.input_video)

	count = -20

	dataset = LoadImages(opt.input_video)

	frame_id = 0
	person_results, veh_results, trajectories = [], [], []
	with tqdm(total=len(dataset)) as pbar:
		# Run inference
		for path, im0s, vid_cap, s in dataset:
			pbar.set_description("Processing %s" % path)

			# Inference

			################################################ video object detection ################################################
			if opt.person_detection.enabled:
				person_results = person_detector.process_frame(im0s)
			if opt.veh_detection.enabled:
				veh_results = veh_detector.process_frame(im0s, veh_classes)

			################################################ video anomaly detection (image) ################################################
			if opt.img_anomaly_detection.enabled:
				anomaly_detector.process_frame(im0s)

			################################################ video object tracking ################################################
			if opt.tracking.enabled:
				online_tlwhs, online_ids = tracker.process_frame(im0s)
				trajectories.extend(
					[
						(frame_id, track_id, *tlwh)
						for track_id, tlwh in zip(online_ids, online_tlwhs)
					]
				)

			################################################ visualization (detection/tracking) ################################################
			imc = im0s.copy()
			if opt.person_detection.enabled:
				imc = plot_boxes(person_results + veh_results, imc)
				if opt.produce_files.enable:
					video_output.write_objects(im0s, imc)
			if opt.tracking.enabled:
				imc = plot_tracking(imc, online_tlwhs, online_ids, frame_id=frame_id)
				if opt.produce_files.enable:
					video_output.write_tracking(imc)
			
			cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
			cv2.setWindowProperty(
				"frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
			)
			cv2.imshow("frame", imc)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				cv2.destroyAllWindows()
				vid_cap.release()
				break
				
			if opt.use_database:
				frame_time = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
				if person_results:
					count_label = person_detector.count_label(person_results, "0")
					influx.insertHumans(opt.input_name, count_label, frame_time)
				if veh_results:
					influx.insertObjects(
						opt.input_name, opt.input_video, veh_results, frame_time
					)
     
			if opt.produce_files.enable:
				# write JSON
				frame_time = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
				if person_results:
					count_label = person_detector.count_label(person_results, person_index)
					json_output.add_humans(opt.input_name, count_label, frame_time)
				if veh_results:
					for label in veh_classes:
						count_label = veh_detector.count_label(veh_results, label)
						label_name = veh_detector.cls2label(label)
						json_output.add_vehicles(opt.input_name, opt.input_video, label_name, count_label, frame_time)

			frame_id += 1
			pbar.update(1)

	################################################ video anomaly detection (image) ################################################
	if opt.img_anomaly_detection.enabled:
		anomaly_scores = anomaly_detector.measure_anomaly_scores()
		max_pred_err = max([pred_err.data.cpu().numpy().mean(0).max() for pred_err in anomaly_detector.pred_err_buffer])
		max_norm_err = max([norm_err.data.cpu().numpy().mean(0).max() for norm_err in anomaly_detector.norm_err_buffer])

	################################################ video anomaly detection (trajectory) ################################################
	if opt.traj_anomaly_detection.enabled:
		scene_df = load_and_window_MT(
			trajectories=trajectories,
			scene=opt.input_name,
			step=5,
			window_size=20,
			stride=20,
		)
		obs_len = opt.traj_anomaly_detection.obs_len
		pred_len = opt.traj_anomaly_detection.pred_len

	dataset = LoadImages(opt.input_video)
	frame_id = 0
	for i, (path, im0s, vid_cap, s) in enumerate(dataset):

		################################################ video anomaly detection (image) ################################################
		if opt.img_anomaly_detection.enabled:
			score_idx = i - (anomaly_detector.t_length - 1)
			anomaly_score = anomaly_scores[score_idx] if score_idx >= 0 else -1
			pred_err = anomaly_detector.pred_err_buffer[frame_id].data.cpu().numpy()
			norm_err = anomaly_detector.norm_err_buffer[frame_id].data.cpu().numpy()
			img = plot_anomaly(im0s, anomaly_score, frame_id=frame_id)
			pred_err_img = plot_pred_err(im0s, pred_err, max_pred_err, opt.img_anomaly_detection.w, opt.img_anomaly_detection.h)
			recon_err_img = plot_norm_err(im0s, norm_err, max_norm_err, opt.img_anomaly_detection.w, opt.img_anomaly_detection.h)
   
			if opt.produce_files.enable:
				frame_time = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
				json_output.add_anomaly(opt.input_name, anomaly_score, opt.input_video, frame_time)
				video_output.write_anomaly(img, pred_err_img, recon_err_img)

			################################################ visualization (anomaly) ################################################
			# concatenate image horizontally
			hstack = np.concatenate((pred_err_img, img, recon_err_img), axis=1)
			# concatenate image vertically
			# vstack = np.concatenate((pred_err_img, img, recon_err_img), axis=0)
			cv2.namedWindow('anomaly', cv2.WINDOW_NORMAL)
			cv2.setWindowProperty('anomaly', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
			cv2.imshow('anomaly', hstack)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				cv2.destroyAllWindows()
				vid_cap.release()
				break

		################################################ video anomaly detection (trajectory) ################################################
		if opt.traj_anomaly_detection.enabled:
			if frame_id in scene_df["frame"].values:
				trajectories = []
				track_ids = scene_df.loc[(scene_df["frame"] == frame_id)]["trackId"].values
				for track_id in track_ids:
					obs_df = scene_df.loc[
						(scene_df["frame"] <= frame_id) & (scene_df["trackId"] == track_id)
					]
					pred_df = scene_df.loc[
						(scene_df["frame"] > frame_id) & (scene_df["trackId"] == track_id)
					]
					if len(obs_df) >= obs_len and len(pred_df) >= pred_len:
						obs_traj = (
							obs_df[["x", "y"]]
							.to_numpy()
							.astype("float32")[-obs_len :]
							.reshape(obs_len, 2)
						)
						pred_traj = (
							pred_df[["x", "y"]]
							.to_numpy()
							.astype("float32")[: pred_len]
							.reshape(pred_len, 2)
						)
						trajectories.append(np.concatenate([obs_traj, pred_traj]))
				if trajectories:
					(
						future_samples_list,
						waypoint_samples_list,
					) = traj_anomaly_detector.process_frame(
						im0s, torch.Tensor(trajectories), opt.input_name
					)
					val_ADE, val_FDE = traj_anomaly_detector.measure_anomaly_score(
						torch.Tensor(trajectories),
						future_samples_list,
						waypoint_samples_list,
					)
					print(f"val_ADE: {val_ADE} - val_FDE: {val_FDE}")
					traj_anomaly_detector.plot_results(im0s, torch.Tensor(trajectories), future_samples_list)

		if opt.use_database:
			frame_time = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
			influx.insertAnomaly(opt.input_name, float(anomaly_score), opt.input_video, frame_time)

		frame_id += 1

	cv2.destroyAllWindows()
	if opt.produce_files.enable:
		json_output.close()
		video_output.release_all()


if __name__ == "__main__":
	opt = opts().init()
	main(opt)
