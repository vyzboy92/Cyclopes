import jetson.inference
import jetson.utils
from imutils.video import WebcamVideoStream, FileVideoStream
import time
import cv2

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
vs = FileVideoStream("../vid.mp4").start()
time.sleep(0.5)
display = jetson.utils.glDisplay()

while display.IsOpen():
    frame = vs.read()
    h, w = frame.shape[:2]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

    cuda_frame = jetson.utils.cudaFromNumpy(frame)

    detections = net.Detect(cuda_frame, w, h)
    display.RenderOnce(cuda_frame, w, h)
    display.SetTitle("Object Detection | Network {:.0g} FPS".format(net.GetNetworkFPS()))
vs.stop()