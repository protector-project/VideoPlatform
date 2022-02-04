"""
Plotting utils
"""

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
import cv2
import skimage.io
from skimage.transform import rescale


def get_color(idx):
	idx = idx * 3
	color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

	return color


def plot_boxes(results, im0, color=(128, 128, 128), txt_color=(255, 255, 255), lw=3):
	"""
	Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
	:param results: contains labels and coordinates predicted by model on the given frame.
	:param frame: Frame which has been scored.
	:return: Frame with bounding boxes and labels ploted on it.
	"""
	h, w = im0.shape[:2]
	for label, *xyxy, conf in results:
		# xyxy = xywh2xyxy(torch.tensor(xywh).view(1, 4))
		p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
		cv2.rectangle(im0, p1, p2, color=color, thickness=lw, lineType=cv2.LINE_AA)
		tf = max(lw - 1, 1)  # font thickness
		# label = self.cls2label(cls)
		w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[
			0
		]  # text width, height
		outside = p1[1] - h - 3 >= 0  # label fits outside box
		p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
		cv2.rectangle(im0, p1, p2, color, -1, cv2.LINE_AA)  # filled
		cv2.putText(
			im0,
			label,
			(p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
			0,
			lw / 3,
			txt_color,
			thickness=tf,
			lineType=cv2.LINE_AA,
		)

	return im0


def plot_anomaly(
	im0,
	anomaly_score,
	anomaly_thres=0.5,
	font=cv2.FONT_HERSHEY_SIMPLEX,
	font_scale=1,
	thickness=2,
	frame_id=0,
):
	im = np.ascontiguousarray(np.copy(im0))

	if anomaly_score >= 0:
		text_scale = max(font_scale, im0.shape[1] / 1600.0)
		top_left_corner_of_text = (0, int(30 * text_scale))

		anomaly_description = "Anomalous" if anomaly_score < anomaly_thres else "Normal"
		color = (0, 0, 255) if anomaly_description == "Anomalous" else (0, 255, 0)

		cv2.putText(
			im,
			f"frame: {frame_id} event: {anomaly_description} score: {anomaly_score:.2f}",
			top_left_corner_of_text,
			font,
			text_scale,
			color,
			thickness=thickness,
		)

	return im


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0.0, ids2=None):
	im = np.ascontiguousarray(np.copy(image))
	im_h, im_w = im.shape[:2]

	top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

	text_scale = max(1, image.shape[1] / 1600.0)
	text_thickness = 2
	line_thickness = max(1, int(image.shape[1] / 500.0))

	radius = max(5, int(im_w / 140.0))
	cv2.putText(
		im,
		"frame: %d num: %d" % (frame_id, len(tlwhs)),
		(0, int(30 * text_scale)),
		cv2.FONT_HERSHEY_SIMPLEX,
		text_scale,
		(0, 0, 255),
		thickness=2,
	)

	for i, tlwh in enumerate(tlwhs):
		x1, y1, w, h = tlwh
		intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
		obj_id = int(obj_ids[i])
		id_text = "{}".format(int(obj_id))
		if ids2 is not None:
			id_text = id_text + ", {}".format(int(ids2[i]))
		_line_thickness = 1 if obj_id <= 0 else line_thickness
		color = get_color(abs(obj_id))
		cv2.rectangle(
			im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
		)
		cv2.putText(
			im,
			id_text,
			(intbox[0], intbox[1] + 30),
			cv2.FONT_HERSHEY_PLAIN,
			text_scale,
			(0, 0, 255),
			thickness=text_thickness,
		)
	return im

def plot_trajectories(gt_future, future_samples, observed, im, resize, with_bg=True, save_path=None):
	plt.scatter(gt_future.detach().cpu()[0,:,0] / resize, gt_future.detach().cpu()[0,:,1] / resize, label='ground truth', zorder=3)
	plt.scatter(future_samples.detach().cpu()[:,0,:,0] / resize, future_samples.detach().cpu()[:,0,:,1] / resize, label='predictions', alpha=0.1, zorder=2)
	plt.scatter(observed[:,0] / resize, observed[:,1] / resize, label='observed_past', color='cyan', zorder=1)
	# scene_image_rescaled = rescale(scene_image.detach().cpu().squeeze()[1].squeeze(), 1/resize)
	# im_rescaled = rescale(im.detach().cpu().squeeze()[1].squeeze(), 1/resize)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # BGR to RGB
	im = np.ascontiguousarray(np.copy(im))
	im_rescaled = rescale(im, 1/resize)
	# plt.imshow(scene_image_rescaled, alpha=0.001)
	if with_bg:
		# plt.imshow(scene_image_rescaled)
		plt.imshow(im_rescaled)
	plt.legend()							

	if save_path is not None:
		plt.savefig(save_path)
	plt.axis('off')
	plt.show()
	plt.savefig("test.png", bbox_inches='tight')