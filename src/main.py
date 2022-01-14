
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os.path
import cv2
import argparse
import torch

from lib.object_detector.object_detector import ObjectDetector
from lib.anomaly_detector.anomaly_detector import AnomalyDetector
from lib.influx.influx_client import InfluxClient
from lib.utils.datasets.dataset_yolo import LoadImages

from lib.utils.torch_utils import select_device


@torch.no_grad()
def run(
    input_filepath,
    input_name,
    anomaly_model,
    detection_model,
    device,
    database_host,
    database_port,
    database_name,
):
    if not os.path.isfile(input_filepath):
        print("Video File Not Found")
        exit()

    if not os.path.isfile(anomaly_model):
        print("Anomaly Model File Not Found")
        exit()

    if not os.path.isfile(detection_model):
        print("Object Detection Model File Not Found")
        exit()

    USE_DATABASE = False

    # Load model
    device = select_device(device)
    # a = AnomalyDetector("pre_trained_models\mpn_piazza_2_sett_3_last.pt")
    # a = AnomalyDetector(args.anomaly_model)
    object_detector = ObjectDetector(detection_model, device)

    if USE_DATABASE:
        i = InfluxClient(
            host=database_host,
            port=int(database_port),
            database=database_name,
        )
        i.createConnection()

    count = -20

    ### Video Info
    # video_file = "video_samples/rec-piazza-fiera-1-20210930T0730-300-mjpeg.avi"
    video_file = input_filepath
    # cam_name = "piazza-fiera"
    cam_name = input_name

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

    dataset = LoadImages(video_file)

    # Run inference
    for path, im, im0s, vid_cap, s in dataset:
        # Inference
        results = object_detector.predict(im, im0s.shape)
        img = object_detector.plot_boxes(results, im0s)
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-I",
        "--input_filepath",
        type=str,
        required=True,
        help="The path to the video file to process",
    )
    parser.add_argument(
        "-N",
        "--input_name",
        type=str,
        required=True,
        help="The name of the place of the video",
    )
    parser.add_argument(
        "-AM",
        "--anomaly_model",
        type=str,
        required=True,
        help="The path to the anomaly model",
    )
    parser.add_argument(
        "-OM",
        "--detection_model",
        type=str,
        required=True,
        help="The path to the object detection model",
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "-DH",
        "--database_host",
        type=str,
        required=False,
        help="IP address of the database",
    )
    parser.add_argument(
        "-DP",
        "--database_port",
        type=str,
        required=False,
        help="Port number of the database",
    )
    parser.add_argument(
        "-DN",
        "--database_name",
        type=str,
        required=False,
        help="Name of the Database",
    )
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
