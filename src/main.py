from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import enum
from pathlib import Path

from pyrsistent import optional

import numpy as np

import _init_paths

import os.path
import cv2
import torch
import influxdb
import pandas as pd
import datetime

from tqdm import tqdm
from lib.anomaly_detector.action_anomaly_detector import ActionAnomalyDetector
from lib.anomaly_detector.traj_anomaly_detector import TrajAnomalyDetector

from lib.object_detector.object_detector import ObjectDetector
from lib.anomaly_detector.anomaly_detector import AnomalyDetector
from lib.influx.influx_client import InfluxClient
from lib.influx.json_influx import InfluxJson
from lib.influx.video_output import VideoOutput
from lib.datasets.dataset import LoadImages
from lib.opts import get_project_root, opts
from lib.tracker.strong_sort_tracker import Tracker
from lib.utils.general import xyxy2xywh
from lib.utils.preprocessing import load_and_window_MT

from lib.utils.torch_utils import select_device
from lib.utils.visualization import plot_actions, plot_anomaly, plot_boxes, plot_gradcam, plot_pred_err, plot_norm_err, plot_tracking, plot_trajectories


OBJECTS_LABELS = ['person', 'bicycle', 'car', 'van', 'truck', 'bus', 'motor']


@torch.no_grad()
def main(opt):
	# Load models
	device = select_device(opt.device)
	# names = []
	if opt.img_anomaly_detection.enabled:
		anomaly_detector = AnomalyDetector(opt.img_anomaly_detection, device)
	if opt.object_detection.enabled:
		object_detector = ObjectDetector(opt.object_detection, device)
		names = object_detector.model.names
	if opt.tracking.enabled:
		tracker = Tracker(opt.tracking, device)
		prev_frame = None
	if opt.traj_anomaly_detection.enabled:
		traj_anomaly_detector = TrajAnomalyDetector(opt.traj_anomaly_detection, device)
	if opt.action_anomaly_detection.enabled:
		action_anomaly_detector = ActionAnomalyDetector(opt.action_anomaly_detection, device)

	if opt.use_database:
		influx = InfluxClient(
			host=opt.database_host,
			port=int(opt.database_port),
			database=opt.database_name,
		)
		influx.createConnection()
  
	if opt.produce_files.enable:
		video_output = VideoOutput(opt)
		try:
			json_output = InfluxJson(opt)
		except ValueError:
			print(f"filename {opt.input_video} not correctly formatted")
			exit()

	count = -20

	dataset = LoadImages(opt.input_video)

	frame_id = 0
	detections = []
	trajectories = []
	
	with tqdm(total=len(dataset)) as pbar:
		# Run inference
		for path, im0s, vid_cap, s in dataset:
			pbar.set_description("Processing %s" % path)

			# Inference
			frame_dets = []

			################################################ video object detection ################################################
			opt.object_detection.enabled = True
			if opt.object_detection.enabled:
				frame_dets = object_detector.process_frame(im0s)
				frame_dets = frame_dets[0]
    
			detections.append(frame_dets)
			frame_dets = []
			# opt.object_detection.enabled = False

			################################################ video anomaly detection (image) ################################################
			if opt.img_anomaly_detection.enabled:
				anomaly_detector.process_frame(im0s)

			################################################ video object tracking ################################################
			if opt.tracking.enabled:
				# online_tlwhs, online_ids = tracker.process_frame(im0s)
				# x1, y1, x2, y2, track_id, class_id, conf
				outputs = tracker.process_frame(im0s, prev_frame, frame_dets)
				trajectories.extend(
					[
						(frame_id, track_id, *xyxy2xywh(np.array(xyxy).reshape(1, 4)).reshape(-1))
						for *xyxy, track_id, class_id, conf in outputs
					]
				)
				prev_frame = im0s

			################################################ video anomaly detection (action) ################################################
			if opt.action_anomaly_detection.enabled:
				with torch.set_grad_enabled(True):
					action_anomaly_detector.process_frame(im0s)

			################################################ visualization (detection/tracking) ################################################
			# video_output.write_original(im0s)
			if opt.object_detection.enabled:
				imc = plot_boxes(frame_dets, im0s, names)
				if opt.produce_files.enable:
					video_output.write_objects(imc)
			if opt.tracking.enabled:
				# imc = plot_tracking(imc, online_tlwhs, online_ids, frame_id=frame_id)
				imc = plot_tracking(im0s, outputs, frame_id=frame_id)
				if opt.produce_files.enable:
					video_output.write_tracking(imc)
			
			# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
			# cv2.setWindowProperty(
			# 	"frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
			# )
			# cv2.imshow("frame", imc)
			# if cv2.waitKey(1) & 0xFF == ord("q"):
			# 	cv2.destroyAllWindows()
			# 	vid_cap.release()
			# 	break
				
			if opt.use_database:
				frame_time = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
				if len(frame_dets):
					person_count = object_detector.count_label(frame_dets, 'person')
					influx.insertHumans(opt.input_name, person_count, frame_time)
					influx.insertObjects(opt.input_name, opt.input_video, frame_dets, frame_time, names)
	 
			if opt.produce_files.enable:
				# write JSON
				frame_time = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
				if len(frame_dets):
					person_count = object_detector.count_label(frame_dets, 'person')
					json_output.add_humans(opt.input_name, person_count, frame_time)
					for label in OBJECTS_LABELS:
						if label != 'person':
							label_count = object_detector.count_label(frame_dets, label)
							json_output.add_vehicles(opt.input_name, opt.input_video, label, label_count, frame_time)

			frame_id += 1
			pbar.update(1)
		
	################################################ video anomaly detection (action) ################################################
	if opt.action_anomaly_detection.enabled:
		anomaly_scores = action_anomaly_detector.anomaly_scores

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

	# base = os.path.basename(opt.input_video)
	# filename = f"output/{os.path.splitext(base)[0]}-vstack6.mp4"
	# hvideo = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 12.5, (1600*3,1200))
	# vvideo = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 12.5, (1600,1200*3))
	# h0, w0 = 1200, 1600
	# h, w = 256, 256
	# r = h / float(h0)
	# # r = w / float(w0)
	# dim = (int(w0 * r), h)
	# # dim = (w, int(h0 * r))
	# video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 12.5, (256*2+dim[0],256))
	# # video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 12.5, (256,(256*2+dim[1])))

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
			# img = cv2.resize(img, dim)
			pred_err_img = plot_pred_err(im0s, pred_err, max_pred_err, opt.img_anomaly_detection.w, opt.img_anomaly_detection.h)
			recon_err_img = plot_norm_err(im0s, norm_err, max_norm_err, opt.img_anomaly_detection.w, opt.img_anomaly_detection.h)
   
			if opt.produce_files.enable:
				frame_time = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
				json_output.add_anomaly(opt.input_name, anomaly_score, opt.input_video, frame_time)
				video_output.write_anomaly(img, pred_err_img, recon_err_img)

			# cv2.imwrite(f"output/pred_err_{frame_id:04d}.png", pred_err_img)
			# cv2.imwrite(f"output/recon_err_{frame_id:04d}.png", recon_err_img)

			################################################ visualization (anomaly) ################################################
			# concatenate image horizontally
			# hstack = np.concatenate((pred_err_img, img, recon_err_img), axis=1)
			# video.write(hstack)
			# concatenate image vertically
			# vstack = np.concatenate((pred_err_img, img, recon_err_img), axis=0)
			# cv2.namedWindow('anomaly', cv2.WINDOW_NORMAL)
			# cv2.setWindowProperty('anomaly', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
			# cv2.imshow('anomaly', hstack)
			# if cv2.waitKey(1) & 0xFF == ord("q"):
			# 	cv2.destroyAllWindows()
			# 	vid_cap.release()
			# 	break

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

		################################################ video anomaly detection (action) ################################################
		if opt.action_anomaly_detection.enabled:
			if frame_id < len(anomaly_scores):
				anomaly_score = anomaly_scores[frame_id]
				y_values = action_anomaly_detector.y_values
				y_indices = action_anomaly_detector.y_indices
				action = action_anomaly_detector.classes_all[y_indices[frame_id][0]][1]
				gradcam_images = action_anomaly_detector.gradcam_images
				img = plot_actions(im0s, y_indices, y_values, action_anomaly_detector.classes_all, f_idx=frame_id)
				frame_dets = detections[frame_id]
				cam_image = plot_gradcam(im0s, gradcam_images, frame_dets, f_idx=frame_id)

			if opt.produce_files.enable:
				frame_time = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
				json_output.add_anomaly(opt.input_name, anomaly_score, action, opt.input_video, frame_time)
				video_output.write_actions(img, cam_image)

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
