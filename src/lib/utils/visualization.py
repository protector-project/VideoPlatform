"""
Plotting utils
"""

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import numpy as np
import cv2
import skimage.io
from skimage.transform import rescale
from pytorch_grad_cam.utils.image import show_cam_on_image


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def plot_boxes(results, im0, names, color=(128, 128, 128), txt_color=(255, 255, 255), lw=3):
	"""
	Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
	:param results: contains labels and coordinates predicted by model on the given frame.
	:param frame: Frame which has been scored.
	:return: Frame with bounding boxes and labels ploted on it.
	"""
	im = np.ascontiguousarray(np.copy(im0))
	h, w = im.shape[:2]
	for *xyxy, conf, cls in results:
		# xyxy = xywh2xyxy(torch.tensor(xywh).view(1, 4))
		p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
		c = int(cls)  # integer class
		color = colors(c, True)
		cv2.rectangle(im, p1, p2, color=color, thickness=lw, lineType=cv2.LINE_AA)
		tf = max(lw - 1, 1)  # font thickness
		label = names[int(cls)]
		w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[
			0
		]  # text width, height
		outside = p1[1] - h - 3 >= 0  # label fits outside box
		p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
		cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
		cv2.putText(
			im,
			label,
			(p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
			0,
			lw / 3,
			txt_color,
			thickness=tf,
			lineType=cv2.LINE_AA,
		)

	return im


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

		anomaly_text = "Anomalous" if anomaly_score < anomaly_thres else "Normal"
		color = (0, 0, 255) if anomaly_text == "Anomalous" else (0, 255, 0)

		cv2.putText(
			im,
			f"frame: {frame_id} event: {anomaly_text} score: {anomaly_score:.2f}",
			top_left_corner_of_text,
			font,
			text_scale,
			color,
			thickness=thickness,
		)

	return im


def plot_pred_err(im0, pred_err, max_val, resize_width, resize_height):
	imc = np.ascontiguousarray(np.copy(im0))
	pred_err = pred_err.mean(0)
	pred_err *= 255.0/max_val
	pred_err = pred_err.astype(np.uint8)
	# th = cv2.threshold(pred_err, 127, 255, cv2.THRESH_BINARY)[1]
	# th = cv2.adaptiveThreshold(pred_err,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	# blur = cv2.GaussianBlur(th, (13,13), 11)
	# pred_err = np.expand_dims(pred_err, axis=0)
	# pred_err = np.transpose(pred_err, (1, 2, 0))
	img_resized = cv2.resize(imc, (resize_width, resize_height))
	# heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
	heatmap_img = cv2.applyColorMap(pred_err, cv2.COLORMAP_JET)
	im = cv2.addWeighted(heatmap_img, 0.7, img_resized, 0.3, 0)
	h, w, c = imc.shape
	im = cv2.resize(im, (w, h))
	return im


def plot_norm_err(im0, norm_err, max_val, resize_width, resize_height):
	imc = np.ascontiguousarray(np.copy(im0))
	norm_err = norm_err.mean(0)
	norm_err *= 255.0/max_val
	norm_err = norm_err.astype(np.uint8)
	# th = cv2.threshold(norm_err, 127, 255, cv2.THRESH_BINARY)[1]
	# th = cv2.adaptiveThreshold(norm_err,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	# blur = cv2.GaussianBlur(th, (13,13), 11)
	# norm_err = np.expand_dims(norm_err, axis=0)
	# norm_err = np.transpose(norm_err, (1, 2, 0))
	img_resized = cv2.resize(imc, (resize_width, resize_height))
	# heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
	heatmap_img = cv2.applyColorMap(norm_err, cv2.COLORMAP_JET)
	im = cv2.addWeighted(heatmap_img, 0.7, img_resized, 0.3, 0)
	h, w, c = imc.shape
	im = cv2.resize(im, (w, h))
	return im


def plot_tracking(image, outputs, frame_id=0, fps=0.0, ids2=None):
	im = np.ascontiguousarray(np.copy(image))
	im_h, im_w = im.shape[:2]
	text_scale = max(1, image.shape[1] / 1600.0)
	text_thickness = 2
	line_thickness = max(1, int(image.shape[1] / 500.0))

	radius = max(5, int(im_w / 140.0))
	cv2.putText(
		im,
		"frame: %d num: %d" % (frame_id, len(outputs)),
		(0, int(30 * text_scale)),
		cv2.FONT_HERSHEY_SIMPLEX,
		text_scale,
		(0, 0, 255),
		thickness=2,
	)

	for j, output in enumerate(outputs):
		bboxes = output[0:4]
		id = output[4]
		cls = output[5]

		bbox_left = output[0]
		bbox_top = output[1]
		bbox_right = output[2]
		bbox_bottom = output[3]
		intbox = tuple(map(int, (bbox_left, bbox_top, bbox_right, bbox_bottom)))
		id_text = "{}".format(int(id))
		c = abs(id)  # integer class
		color = colors(c, True)
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


# def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0.0, ids2=None):
# 	im = np.ascontiguousarray(np.copy(image))
# 	im_h, im_w = im.shape[:2]

# 	top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

# 	text_scale = max(1, image.shape[1] / 1600.0)
# 	text_thickness = 2
# 	line_thickness = max(1, int(image.shape[1] / 500.0))

# 	radius = max(5, int(im_w / 140.0))
# 	cv2.putText(
# 		im,
# 		"frame: %d num: %d" % (frame_id, len(tlwhs)),
# 		(0, int(30 * text_scale)),
# 		cv2.FONT_HERSHEY_SIMPLEX,
# 		text_scale,
# 		(0, 0, 255),
# 		thickness=2,
# 	)

# 	for i, tlwh in enumerate(tlwhs):
# 		x1, y1, w, h = tlwh
# 		intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
# 		obj_id = int(obj_ids[i])
# 		id_text = "{}".format(int(obj_id))
# 		if ids2 is not None:
# 			id_text = id_text + ", {}".format(int(ids2[i]))
# 		_line_thickness = 1 if obj_id <= 0 else line_thickness
# 		color = get_color(abs(obj_id))
# 		cv2.rectangle(
# 			im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
# 		)
# 		cv2.putText(
# 			im,
# 			id_text,
# 			(intbox[0], intbox[1] + 30),
# 			cv2.FONT_HERSHEY_PLAIN,
# 			text_scale,
# 			(0, 0, 255),
# 			thickness=text_thickness,
# 		)
# 	return im


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


def plot_actions(im0, pred_indices, pred_values, classes, f_idx):
	imc = np.ascontiguousarray(np.copy(im0))
	position = (0, 30)
	pred_values_list = pred_values[f_idx]
	pred_indices_list = pred_indices[f_idx]
	text = [f"{classes[idx][1]}: {score:.6f}" for idx, score in zip(pred_indices_list, pred_values_list)]
	text = "\n".join(text)
	font_scale = 0.8
	thickness=2
	font = cv2.FONT_HERSHEY_SIMPLEX
	line_type = cv2.LINE_AA
	
	text_color_bg = (255, 255, 255)
	text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
	text_w, text_h = text_size
	line_height = text_h + 5
	x, y0 = position
	rect_w = max(cv2.getTextSize(x, font, font_scale, thickness)[0][0] for x in text.split("\n"))
	rect_h = line_height * len(text.split("\n")) - 5
	cv2.rectangle(imc, (x, y0 - line_height), (x + rect_w, y0 + rect_h), text_color_bg, -1)
	
	for i, line in enumerate(text.split("\n")):
		y = y0 + i * line_height
		c = int(pred_indices_list[i])  # integer class
		color = colors(c, True)
		cv2.putText(
			imc,
			line,
			(x, y),
			font,
			font_scale,
			color,
			thickness,
			line_type
		)
		
	return imc

def plot_gradcam(im0, gradcam_images, frame_dets, f_idx):
	grayscale_cam = gradcam_images[f_idx]
 
	if len(frame_dets):
		mask = np.zeros(im0.shape[:2], dtype=np.uint8)
		for *xyxy, _, _ in frame_dets:
			p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
			cv2.rectangle(mask, p1, p2, (255,255,255), -1)
	else:
		mask = np.ones(im0.shape[:2], dtype=np.uint8) * 255
   
	mask = np.ascontiguousarray(np.copy(mask))
	mask = cv2.resize(mask, (224, 224))
	mask = np.float32(mask) / 255
   
   	# Mask input image with binary mask
	# grayscale_cam = cv2.bitwise_and(grayscale_cam, mask)
	grayscale_cam *= mask
  
	imc = np.ascontiguousarray(np.copy(im0))
	imc = cv2.resize(imc, (224, 224))
	imc = np.float32(imc) / 255
	cam_image = show_cam_on_image(imc, grayscale_cam)

	return cam_image


def plot_traj_anomaly(image, results):
	im = np.ascontiguousarray(np.copy(image))
	text_scale = max(1, image.shape[1] / 1600.0)
	text_thickness = 2
	line_thickness = max(1, int(image.shape[1] / 500.0))

	for j, output in enumerate(results):
		text = output[5]
		score = output[6]
		bbox_left = output[1]
		bbox_top = output[2]
		bbox_right = output[1]+output[3]
		bbox_bottom = output[2]+output[4]
		intbox = tuple(map(int, (bbox_left, bbox_top, bbox_right, bbox_bottom)))

		if "ANOMALY" in text:
			color = (0, 0, 255)
		else:
			color = (0, 255, 0)

		cv2.rectangle(
			im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
		)
		cv2.putText(
			im,
			f"{text}-{score}",
			(intbox[0], intbox[1] + 30),
			cv2.FONT_HERSHEY_PLAIN,
			text_scale,
			color,
			thickness=text_thickness,
		)
	return im
