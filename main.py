import cv2
from vidgear.gears import CamGear

from AnomalyDetector import AnomalyDetector
from ObjectDetector import ObjectDetector
from InfluxClient import InfluxClient

USE_DATABASE = True

a = AnomalyDetector()
o = ObjectDetector()

if USE_DATABASE:
    i = InfluxClient(host="192.168.15.95", port=8086, database="protector")
    i.createConnection()


count = -20

video_file = "video_samples/rec-piazza-fiera-1-20210930T0730-300-mjpeg.avi"
cam_name = "piazza-fiera"
vidcap = CamGear(source=video_file).start()
img = vidcap.read()


while img is not None:
    a.processFrame(img)
    if count > 0:
        r = a.lastResult()
        # r = a.previewResults()
        img = a.plotAnomaly(img, r[-1])
        if USE_DATABASE:
            i.insertAnomaly(cam_name, r[-1], "None")
    count += 1
    results = o.score_frame(img)
    if USE_DATABASE:
        count_label = o.count_label(results, "person")
        i.insertHumans(cam_name, count_label)
        i.insertObjects(cam_name, "None", results, o.class_to_label)
    frame = o.plot_boxes(results, img)
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    img = vidcap.read()

vidcap.stop()
