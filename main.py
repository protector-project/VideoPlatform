import cv2

from AnomalyDetector import AnomalyDetector
from InfluxClient import InfluxClient
from ObjectDetector import ObjectDetector

USE_DATABASE = True

a = AnomalyDetector()
o = ObjectDetector()

if USE_DATABASE:
    i = InfluxClient(host="192.168.15.95", port=8086, database="protector")
    i.createConnection()

count = -20

### Video Info
video_file = "video_samples/rec-piazza-fiera-1-20210930T0730-300-mjpeg.avi"
cam_name = "piazza-fiera"

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
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    frame_exists, img = cap.read()

cv2.destroyAllWindows()
cap.release()
