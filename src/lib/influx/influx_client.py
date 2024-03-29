from influxdb import InfluxDBClient


class InfluxClient:
    def __init__(self, host, port, database):
        self.host = host
        self.port = port
        self.database = database
        self.client = None

    def createConnection(self):
        try:
            self.client = InfluxDBClient(
                host=self.host, port=self.port, database=self.database
            )
        except Exception as e:
            print(e)

    def insertAnomaly(self, cameraName, value, clipName, timestamp):
        service_url = "http://localhost:5000"
        url = service_url + "/" + clipName + "/" + str(timestamp)
        json_body = [
            {
                "measurement": "anomaly",
                "tags": {"camera": cameraName, "file_name": clipName},
                "fields": {"score": value, "video_timestamp": timestamp, "url": url},
            }
        ]
        self.client.write_points(json_body)

    def insertHumans(self, cameraName, value, timestamp):
        json_body = [
            {
                "measurement": "persons",
                "tags": {"camera": cameraName},
                "fields": {"score": value, "video_timestamp": timestamp},
            }
        ]
        self.client.write_points(json_body)

    def insertObjects(self, camera_name, clip_name, detections, timestamp, names):
        json_body = []
        labels = [names[int(cls)] for *xywh, conf, cls in detections]
        count_labels = {i: labels.count(i) for i in labels if i != "person"}
        for k, v in count_labels.items():
            json_body.append(
                {
                    "measurement": "objects",
                    "tags": {
                        "camera": camera_name,
                        "file_name": clip_name,
                        "label": k,
                    },
                    "fields": {"value": v, "video_timestamp": timestamp},
                }
            )
        self.client.write_points(json_body)

    def insertObject(self, cameraName, clipName, labels, boundingBoxes):
        json_body = []
        for index, label in enumerate(labels):
            row = boundingBoxes[index]
            json_body.append(
                {
                    "measurement": "objects",
                    "tags": {
                        "camera": cameraName,
                        "file_name": clipName,
                        "label": label,
                    },
                    "fields": {
                        "value": 1,
                        "x1": row[0],
                        "x2": row[1],
                        "y1": row[2],
                        "y2": row[3],
                    },
                }
            )
        self.client.write_points(json_body)

    def insert_trajectory(self, camera_name, clip_name, trajectories):
        json_body = []
        for *xyxy, track_id, class_id, conf in trajectories:
            bbox = xyxy
            bbox_left = bbox[0]
            bbox_top = bbox[1]
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            json_body.append(
                {
                    "measurement": "trajectories",
                    "tags": {
                        "camera": camera_name,
                        "file_name": clip_name
                    },
                    "fields": {
                        "frame_id": frame_id,
                        "track_id": track_id,
                        "bbox_left": bbox_left,
                        "bbox_top": bbox_top,
                        "bbox_w": bbox_w,
                        "bbox_h": bbox_h,
                        "class_id": class_id,
                        "conf": conf,
                    },
                }
            )
        self.client.write_points(json_body)