from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os.path
import cv2
import torch

from tqdm import tqdm

from lib.object_detector.object_detector import ObjectDetector
from lib.anomaly_detector.anomaly_detector import AnomalyDetector
from lib.influx.influx_client import InfluxClient
from lib.datasets.dataset import LoadImages
from lib.opts import opts
from lib.tracker.tracker import Tracker

from lib.utils.torch_utils import select_device
from lib.utils.visualization import plot_anomaly, plot_boxes, plot_tracking


USE_DATABASE = False


@torch.no_grad()
def main(opt):
    # Load models
    device = select_device(opt.device)
    anomaly_detector = AnomalyDetector(opt, device)
    object_detector = ObjectDetector(opt, device)
    tracker = Tracker(opt, device)

    if USE_DATABASE:
        i = InfluxClient(
            host=opt.database_host,
            port=int(opt.database_port),
            database=opt.database_name,
        )
        i.createConnection()

    count = -20

    ### Video Info
    # video_file = "video_samples/rec-piazza-fiera-1-20210930T0730-300-mjpeg.avi"
    # video_file = opt.input_video
    # cam_name = "piazza-fiera"
    # cam_name = input_name

    # ### Video capture
    # cap = cv2.VideoCapture(video_file)
    # frame_exists, img = cap.read()

    # while frame_exists:
    #     # a.processFrame(img)
    #     # get the current time of the frame
    #     frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    #     # if count > 0:
    #     #     r = a.lastResult()
    #     #     # r = a.previewResults()
    #     #     img = a.plotAnomaly(img, r[-1])
    #     #     if USE_DATABASE:
    #     #         i.insertAnomaly(cam_name, r[-1], video_file, frame_time)
    #     # count += 1
    #     results = o.score_frame(img)
    #     if USE_DATABASE:
    #         count_label = o.count_label(results, "person")
    #         i.insertHumans(cam_name, count_label)
    #         i.insertObjects(cam_name, video_file, results, o.class_to_label, frame_time)
    #     frame = o.plot_boxes(results, img)
    #     # cv2.imshow("image", img)
    #     # key = cv2.waitKey(1) & 0xFF
    #     # if key == ord("q"):
    #     #     break
    #     frame_exists, img = cap.read()

    # cv2.destroyAllWindows()
    # cap.release()

    dataset = LoadImages(opt.input_video)

    frame_id = 0
    # Run inference
    for path, im0s, vid_cap, s in dataset:
        # Inference

        # Video object detection
        results = object_detector.process_frame(im0s)
        img = plot_boxes(results, im0s)

        # Video anomaly detection
        anomaly_detector.process_frame(im0s)

        # Video object tracking
        # online_tlwhs, online_ids = tracker.process_frame(im0s)
        # img = plot_tracking(im0s, online_tlwhs, online_ids, frame_id=frame_id)

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            vid_cap.release()
            break

        frame_id += 1

    anomaly_scores = anomaly_detector.measure_anomaly_scores()
    
    dataset = LoadImages(opt.input_video)

    frame_id = 0
    for i, (path, im0s, vid_cap, s) in enumerate(dataset):
        score_idx = i - (anomaly_detector.t_length - 1)
        anomaly_score = anomaly_scores[score_idx] if score_idx >= 0 else -1
        img = plot_anomaly(im0s, anomaly_score, frame_id=frame_id)

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            vid_cap.release()
            break

        frame_id += 1


if __name__ == "__main__":
    opt = opts().init()
    main(opt)
