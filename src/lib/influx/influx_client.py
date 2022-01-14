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

    def insertHumans(self, cameraName, value):
        json_body = [
            {
                "measurement": "persons",
                "tags": {"camera": cameraName},
                "fields": {"score": value},
            }
        ]
        self.client.write_points(json_body)

    def insertObjects(self, camera_name, clip_name, results, labelfunc, timestamp):
        json_body = []
        labels, cord = results
        real_labels = [labelfunc(l) for l in labels]
        count_labels = {i: real_labels.count(i) for i in real_labels}
        for k, v in count_labels.items():
            json_body.append(
                {
                    "measurement": "objects",
                    "tags": {
                        "camera": camera_name,
                        "file_name": clip_name,
                        "label": k,
                    },
                    "fields": {
                        "value": v,
                        "video_timestamp": timestamp
                    },
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
