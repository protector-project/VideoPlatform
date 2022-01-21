import torch


class VehicleDector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def process_frame(self, im0):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        frame = [im0]
        results = self.model(im0)
        labels, xywh = (
            results.xyxyn[0][:, -1].to("cpu").numpy(),
            results.xyxyn[0][:, :-1].to("cpu").numpy(),
        )
        labels = [self.cls2label(cls) for cls in labels]
        return labels, xywh

    def cls2label(self, cls):
        """
        For a given label value, return corresponding string label.
        :param cls: numeric label
        :return: corresponding string label
        """
        return self.model.classes[int(cls)]

    def count_label(self, results, label):
        n = len([cls == label for cls, *xywh in results])
        return n