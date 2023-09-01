"""import cv2
import numpy

#cv2.namedWindow("preview")
vc = cv2.VideoCapture(-1)

while True:
    rval, frame = vc.read()
    cv2.imshow("preview", frame)
    key = cv2.waitKey(1)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")"""

import cv2
import time
from threading import Thread

contVar = True
cv2.namedWindow("preview", flags=cv2.WINDOW_AUTOSIZE)

class ThreadedCamera(object):
    def __init__(self):
        self.capture = cv2.VideoCapture(-1)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1 / 30
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)

    def show_frame(self):
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(self.FPS_MS)
        if key == 27:  # exit on ESC
            global contVar
            contVar = False
        self.caputure.release()

if __name__ == '__main__':
    threaded_camera = ThreadedCamera()
    while contVar:
        try:
            threaded_camera.show_frame()
        except AttributeError:
            pass
    cv2.destroyWindow("preview")