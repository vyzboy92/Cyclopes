import jetson.inference
import jetson.utils
import sys

net = jetson.inference.detectNet("facenet", threshold=0.5)
camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")
display = jetson.utils.glDisplay()
while display.IsOpen():
	try:
	
		img, width, height = camera.CaptureRGBA()		
		detections = net.Detect(img, width, height)
		display.RenderOnce(img, width, height)		
		display.SetTitle("Object Detection | Network {:.0g} FPS".format(net.GetNetworkFPS()))

	except:
		sys.exit(0)	
	net.PrintProfilerTimes()