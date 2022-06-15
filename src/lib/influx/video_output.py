import cv2
from pathlib import Path
import datetime
import os

class VideoOutput:
    
    '''
    This class is responsable for outputing the frames into video files
    '''
    
    def __init__(self, input_video):
        
        # get video info
        base_file_name = os.path.basename(input_video)
        info = cv2.VideoCapture(input_video)
        x_shape = int(info.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        four_cc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # create video outputs
        original_out = Path(base_file_name).stem + "_original.mp4"
        tracking_out = Path(base_file_name).stem + "_tracking.mp4"
        objects_out = Path(base_file_name).stem + "_objects.mp4"
        anomaly_out = Path(base_file_name).stem + "_anomaly_score.mp4"
        pred_err_out = Path(base_file_name).stem + "_pred_err.mp4"
        recon_err_out = Path(base_file_name).stem + "_recon_err.mp4"
        
        self.original_writer = cv2.VideoWriter("out/{}".format(original_out), four_cc, 12.5, (x_shape, y_shape))
        self.tracking_writer = cv2.VideoWriter("out/{}".format(tracking_out), four_cc, 12.5, (x_shape, y_shape))
        self.objects_writer = cv2.VideoWriter("out/{}".format(objects_out), four_cc, 12.5, (x_shape, y_shape))
        self.anomaly_writer = cv2.VideoWriter("out/{}".format(anomaly_out), four_cc, 12.5, (x_shape, y_shape))
        self.pred_err_writer = cv2.VideoWriter("out/{}".format(pred_err_out), four_cc, 12.5, (x_shape, y_shape))
        self.recon_err_writer = cv2.VideoWriter("out/{}".format(recon_err_out), four_cc, 12.5, (x_shape, y_shape))
        
    def write_objects(self, original_frame, object_frame):
        self.original_writer.write(original_frame)
        self.objects_writer.write(object_frame)
        
    def write_tracking(self, tracking_frame):
        self.tracking_writer.write(tracking_frame)

    def write_anomaly(self, anomaly_score_frame, pred_err_frame, recon_err_frame):
        self.anomaly_writer.write(anomaly_score_frame)
        self.pred_err_writer.write(pred_err_frame)
        self.recon_err_writer.write(recon_err_frame)
        
    def release_all(self):
        self.original_writer.release()
        self.tracking_writer.release()
        self.objects_writer.release()
        self.anomaly_writer.release()
        self.pred_err_writer.release()
        self.recon_err_writer.release()