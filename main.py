import argparse
import os.path

import cv2

from AnomalyDetection.AnomalyDetector import AnomalyDetector
from InfluxClient import InfluxClient
from ObjectDetection.ObjectDetector import ObjectDetector

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
    "-DH",
    "--database_host",
    type=str,
    required=True,
    help="IP address of the database",
)
parser.add_argument(
    "-DP",
    "--database_port",
    type=str,
    required=True,
    help="Port number of the database",
)
parser.add_argument(
    "-DN",
    "--database_name",
    type=str,
    required=True,
    help="Name of the Database",
)

args = parser.parse_args()

if not os.path.isfile(args.input_filepath):
    print("Video File Not Found")
    exit()

if not os.path.isfile(args.anomaly_model):
    print("Anomaly Model File Not Found")
    exit()

USE_DATABASE = True

# a = AnomalyDetector("pre_trained_models\mpn_piazza_2_sett_3_last.pt")
a = AnomalyDetector(args.anomaly_model)
o = ObjectDetector()

if USE_DATABASE:
    i = InfluxClient(
        host=args.database_host,
        port=int(args.database_port),
        database=args.database_name,
    )
    i.createConnection()

count = -20

### Video Info
# video_file = "video_samples/rec-piazza-fiera-1-20210930T0730-300-mjpeg.avi"
video_file = args.input_filepath
# cam_name = "piazza-fiera"
cam_name = args.input_name


### Video capture
cap = cv2.VideoCapture(video_file)
frame_exists, img = cap.read()

while frame_exists:
    a.processFrame(img)
    # get the current time of the frame
    frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    if count > 0:
        r = a.lastResult()
        # r = a.previewResults()
        img = a.plotAnomaly(img, r[-1])
        if USE_DATABASE:
            i.insertAnomaly(cam_name, r[-1], video_file, frame_time)
    count += 1
    results = o.score_frame(img)
    if USE_DATABASE:
        count_label = o.count_label(results, "person")
        i.insertHumans(cam_name, count_label)
        i.insertObjects(cam_name, video_file, results, o.class_to_label, frame_time)
    frame = o.plot_boxes(results, img)
    # cv2.imshow("image", img)
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord("q"):
    #     break
    frame_exists, img = cap.read()

cv2.destroyAllWindows()
cap.release()
