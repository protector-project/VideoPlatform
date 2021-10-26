import cv2
from vidgear.gears import CamGear, WriteGear
from time import time


class FrameFactory:
    
    def __init__(self, videoSource):
        self.source = videoSource

    def createVideoStream(self):
        stream = CamGear(source=self.source).start()
        return stream

    def processBuffer(self, list_of_scores, img_buffer):
        for element in list_of_scores:
            if element["score"] < 22:
                min = element["index"] - 100
                max = element["index"] + 100 
                if min < 0:
                    min = 0
            clipName = "piazza" + "_" + str(time())
            self.createClip(img_buffer[min:max], clipName)


    def createClip(self, frames, clipName):
        out = WriteGear(output_filename="{}.avi".format(clipName))
        for frame in frames:
            out.write(frame)
        out.close()

    def plotBoxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = (
                    int(row[0] * x_shape),
                    int(row[1] * y_shape),
                    int(row[2] * x_shape),
                    int(row[3] * y_shape),
                )
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(
                    frame, labels[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2
                )

        return frame
