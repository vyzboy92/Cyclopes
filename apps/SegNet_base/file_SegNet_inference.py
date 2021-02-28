import jetson.inference
import jetson.utils
from imutils.video import FileVideoStream
import argparse
import ctypes
import sys
import time
import cv2


vs = FileVideoStream("city.mp4").start()
time.sleep(0.5)

initial_frame = vs.read()
height, width = initial_frame.shape[:2]

network = "fcn-resnet18-cityscapes-1024x512"
net = jetson.inference.segNet(network)

net.SetOverlayAlpha(50.0)

frame_overlay = jetson.utils.cudaAllocMapped(width * height * 4 * ctypes.sizeof(ctypes.c_float))

display = jetson.utils.glDisplay()

while display.IsOpen():

	try:

		frame = vs.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

		cuda_frame = jetson.utils.cudaFromNumpy(frame)

		# process the segmentation network
		net.Process(cuda_frame, width, height, "void")

		# generate the overlay and mask
		net.Overlay(frame_overlay, width, height, "linear")

		# render the images
		display.BeginRender()
		display.Render(frame_overlay, width, height)
		display.EndRender()

		# update the title bar
		display.SetTitle("{:s} | Network {:.0f} FPS".format(network, net.GetNetworkFPS()))
	except:
		sys.exit(0)
vs.stop