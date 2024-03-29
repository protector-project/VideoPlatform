import cv2
from pathlib import Path
import datetime
import os

class VideoOutput:
    '''
    This class is responsible for outputing the frames into video files
    '''
    
    def __init__(self, opt):
        
        # get video info
        base_file_name = os.path.basename(opt.input_video)
        info = cv2.VideoCapture(opt.input_video)
        w = int(info.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = info.get(cv2.CAP_PROP_FPS)
        # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writers = []
        
        original_out = Path(base_file_name).stem + "_original.mp4"
        self.original_writer = cv2.VideoWriter(os.path.join(opt.output_root, original_out), fourcc, fps, (w, h))
        self.video_writers.append(self.original_writer)
        
        # create video outputs
        # if opt.person_detection.enabled or opt.veh_detection.enabled:
        if opt.object_detection.enabled:
            objects_out = Path(base_file_name).stem + "_objects.mp4"
            self.objects_writer = cv2.VideoWriter(os.path.join(opt.output_root, objects_out), fourcc, fps, (w, h))
            self.video_writers.append(self.objects_writer)
            
        if opt.tracking.enabled:
            tracking_out = Path(base_file_name).stem + "_tracking.mp4"
            self.tracking_writer = cv2.VideoWriter(os.path.join(opt.output_root, tracking_out), fourcc, fps, (w, h))
            self.video_writers.append(self.tracking_writer)
            
        if opt.img_anomaly_detection.enabled:
            anomaly_out = Path(base_file_name).stem + "_anomaly_score.mp4"
            pred_err_out = Path(base_file_name).stem + "_pred_err.mp4"
            recon_err_out = Path(base_file_name).stem + "_recon_err.mp4"
            self.anomaly_writer = cv2.VideoWriter(os.path.join(opt.output_root, anomaly_out), fourcc, fps, (w, h))    
            self.pred_err_writer = cv2.VideoWriter(os.path.join(opt.output_root, pred_err_out), fourcc, fps, (w, h))
            self.recon_err_writer = cv2.VideoWriter(os.path.join(opt.output_root, recon_err_out), fourcc, fps, (w, h))
            self.video_writers.append(self.anomaly_writer)
            self.video_writers.append(self.pred_err_writer)
            self.video_writers.append(self.recon_err_writer)

        if opt.action_anomaly_detection.enabled:
            actions_out = Path(base_file_name).stem + "_actions.mp4"
            gradcam_out = Path(base_file_name).stem + "_gradcam.mp4"
            self.actions_writer = cv2.VideoWriter(os.path.join(opt.output_root, actions_out), fourcc, fps, (w, h))
            input_size = opt.action_anomaly_detection.data.input_size
            self.gradcam_writer = cv2.VideoWriter(os.path.join(opt.output_root, gradcam_out), fourcc, fps, (input_size, input_size))
            self.video_writers.append(self.actions_writer)
            self.video_writers.append(self.gradcam_writer)

        if opt.traj_anomaly_detection_cluster.enabled:
            traj_anomaly_detection_out = Path(base_file_name).stem + "_traj_anomaly.mp4"
            self.traj_anomaly_detection_writer = cv2.VideoWriter(os.path.join(opt.output_root, traj_anomaly_detection_out), fourcc, fps, (w, h))
            self.video_writers.append(self.traj_anomaly_detection_writer)

    def write_original(self, original_frame):
         self.original_writer.write(original_frame)
        
    def write_objects(self, object_frame):
        self.objects_writer.write(object_frame)
        
    def write_tracking(self, tracking_frame):
        self.tracking_writer.write(tracking_frame)

    def write_anomaly(self, anomaly_score_frame, pred_err_frame, recon_err_frame):
        self.anomaly_writer.write(anomaly_score_frame)
        self.pred_err_writer.write(pred_err_frame)
        self.recon_err_writer.write(recon_err_frame)

    def write_actions(self, actions_frame, cam_image):
        self.actions_writer.write(actions_frame)
        self.gradcam_writer.write(cam_image)

    def write_traj_anomaly_detection(self, frame):
        self.traj_anomaly_detection_writer.write(frame)
        
    def release_all(self):
        for video_writer in self.video_writers:
            video_writer.release()