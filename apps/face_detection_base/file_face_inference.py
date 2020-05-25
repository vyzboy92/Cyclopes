import jetson.inference
import jetson.utils
from imutils.video import WebcamVideoStream, FileVideoStream
import time
import cv2
import sys

net = jetson.inference.detectNet("facenet", threshold=0.5)
vs = FileVideoStream("face.mp4").start()
time.sleep(0.5)
display = jetson.utils.glDisplay()

while display.IsOpen():
    try:
        frame = vs.read()
        h, w = frame.shape[:2]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

        cuda_frame = jetson.utils.cudaFromNumpy(frame)

        detections = net.Detect(cuda_frame, w, h)
        display.RenderOnce(cuda_frame, w, h)
        display.SetTitle("Object Detection | Network {:.0g} FPS".format(net.GetNetworkFPS()))
    except:
        sys.exit(0)
vs.stop()