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
from lib.object_detector.vehicle_detector import VehicleDector
from lib.opts import opts
from lib.tracker.tracker import Tracker

from lib.utils.torch_utils import select_device
from lib.utils.visualization import plot_anomaly, plot_boxes, plot_tracking


USE_DATABASE = True


@torch.no_grad()
def main(opt):
    # Load models
    device = select_device(opt.device)
    anomaly_detector = AnomalyDetector(opt, device)
    object_detector = ObjectDetector(opt, device)
    tracker = Tracker(opt, device)
    vehicle_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    if USE_DATABASE:
        i = InfluxClient(
            host=opt.database_host,
            port=int(opt.database_port),
            database=opt.database_name,
        )
        i.createConnection()

    count = -20

    dataset = LoadImages(opt.input_video)

    frame_id = 0
    # Run inference
    for path, im0s, vid_cap, s in dataset:
        # Inference

        # Video object detection
        results = object_detector.process_frame(im0s)
        img = plot_boxes(results, im0s)

        # Vehicle detection
        vehicles = vehicle_detector.process_frame(im0s)

        # Video anomaly detection
        anomaly_detector.process_frame(im0s)

        # Video object tracking
        online_tlwhs, online_ids = tracker.process_frame(im0s)
        img = plot_tracking(im0s, online_tlwhs, online_ids, frame_id=frame_id)

        # objects visualization
        # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow('frame', img)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     cv2.destroyAllWindows()
        #     vid_cap.release()
        #     break

        if USE_DATABASE:
            frame_time = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if USE_DATABASE:
                print(results)
                if results != []:
                    count_label = object_detector.count_label(results, "0")
                    i.insertHumans(opt.input_name, count_label, frame_time)
                if vehicles != []:
                    i.insertObjects(
                        opt.input_name, opt.input_video, vehicles, frame_time
                    )
        print(frame_time)

        frame_id += 1

    anomaly_scores = anomaly_detector.measure_anomaly_scores()

    dataset = LoadImages(opt.input_video)

    frame_id = 0
    for i, (path, im0s, vid_cap, s) in enumerate(dataset):
        score_idx = i - (anomaly_detector.t_length - 1)
        anomaly_score = anomaly_scores[score_idx] if score_idx >= 0 else -1
        img = plot_anomaly(im0s, anomaly_score, frame_id=frame_id)

        # anomaly visualization
        # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow('frame', img)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     cv2.destroyAllWindows()
        #     vid_cap.release()
        #     break

        if USE_DATABASE:
            frame_time = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            i.insertAnomaly(opt.input_name, anomaly_score, opt.input_video, frame_time)

        frame_id += 1


if __name__ == "__main__":
    opt = opts().init()
    main(opt)
