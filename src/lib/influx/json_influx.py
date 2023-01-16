from pathlib import Path

import datetime
import os

class InfluxJson:
    
    '''
    This Class is responsable for print in the log the json file of
    the objects detect to insert into InfluxDB
    '''
    
    def __init__(self, opt):
        # get video info
        if "mt" in opt.input_video:
            base_file_name = os.path.basename(opt.input_video)
            base_file_name = os.path.splitext(base_file_name)[0]
            time_from_the_name = base_file_name.split("-")[-3]
            self.start_time_video = datetime.datetime.strptime(time_from_the_name, "%Y%m%dT%H%M%S")
            # staged recording
            # time_from_the_name = "".join(base_file_name.split("-")[5:-1])
            # self.start_time_video = datetime.datetime.strptime(time_from_the_name, "%Y%m%dT%H%M%S")
        elif "ei" in opt.input_video:
            base_file_name = os.path.basename(opt.input_video)
            base_file_name = os.path.splitext(base_file_name)[0]
            time_from_the_name = base_file_name.replace("_", "")
            self.start_time_video = datetime.datetime.strptime(time_from_the_name, "%d.%m.%Y г. %H%M%S (%Z%z)")
        elif "pza" in opt.input_video:
            base_file_name = os.path.basename(opt.input_video)
            base_file_name = os.path.splitext(base_file_name)[0]
            time_from_the_name = base_file_name.split("_")[1]
            self.start_time_video = datetime.datetime.strptime(time_from_the_name, "%Y%m%d%H%M%S")
        else:
            raise ValueError
        
        # create outfile
        outputfile_name = base_file_name + "_influx.json"
        self.outputfile = open(os.path.join(opt.output_root, outputfile_name), "w")
        
    
    def add_anomaly(self, camera_name, value, action, clip_name, timestamp):
        current_time = self.start_time_video + datetime.timedelta(seconds=timestamp)
        current_time = current_time.strftime("%m/%d/%Y %H:%M:%S")
        service_url = "https://protector.smartcommunitylab.it/show"
        file_name = Path(clip_name).stem
        url = service_url + "/" + file_name + "/" + str(timestamp)
        json_body = [
            {	
                "time": current_time,
                "measurement": "anomaly",
                "tags": {"camera": camera_name, "file_name": file_name},
                "fields": {"score": value, "action": action, "video_timestamp": timestamp, "url": url},
            }
        ]
        self.outputfile.write(str(json_body))
        self.outputfile.write("\n")
        # print(json_body)


    def add_humans(self, camera_name, value, timestamp):
        current_time = self.start_time_video + datetime.timedelta(seconds=timestamp)
        current_time = current_time.strftime("%m/%d/%Y %H:%M:%S")
        json_body = [
            {	
                "time": current_time,
                "measurement": "humans",
                "tags": {"camera": camera_name},
                "fields": {"score": value, "video_timestamp": timestamp},
            }
        ]
        self.outputfile.write(str(json_body))
        self.outputfile.write("\n")
        # print(json_body)

    
    def add_vehicles(self, camera_name, clip_name, label_veh, n_veh, timestamp):
        current_time = self.start_time_video + datetime.timedelta(seconds=timestamp)
        current_time = current_time.strftime("%m/%d/%Y %H:%M:%S")
        json_body = [
            {
                "time": current_time,
                "measurement": "objects",
                "tags": {
                    "camera": camera_name,
                    "file_name": clip_name,
                    "label": label_veh,
                },
                "fields": {"value": n_veh, "video_timestamp": timestamp},
            }	
        ]
        self.outputfile.write(str(json_body))
        self.outputfile.write("\n")
        # print(json_body)
        
    def add_trajectories(self, camera_name, clip_name, frame_id, trajectories, timestamp):
        current_time = self.start_time_video + datetime.timedelta(seconds=timestamp)
        current_time = current_time.strftime("%m/%d/%Y %H:%M:%S")
        json_body = []
        for *xyxy, track_id, class_id, conf in trajectories:
            bbox = xyxy
            bbox_left = bbox[0]
            bbox_top = bbox[1]
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            json_body.append(
                {
                    "time": current_time,
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
        self.outputfile.write(str(json_body))
        self.outputfile.write("\n")
    
    def close(self):
        self.outputfile.close()