# Project-C Vision Apps
This folder contains experimental apps which enables vision inferencing on the Jetson Nano and TX2 platforms.
Experimental code enables testing and validating vision workflow in Jetson devices. Based on ```jetson-inference``` library by [dusty-nv](https://github.com/dusty-nv/jetson-inference)

## Installation
### jetson-inference

Here's a condensed form of the commands to download, build, and install ```jetson-inference```:

``` bash
$ sudo apt-get update
$ sudo apt-get install git cmake libpython3-dev python3-numpy
$ git clone --recursive https://github.com/dusty-nv/jetson-inference
$ cd jetson-inference
$ mkdir build
$ cd build
$ cmake ../
$ make -j$(nproc)
$ sudo make install
$ sudo ldconfig
```
### imutils
Run command:
```pip install imutils```

## Apps
Each app contains python samples for running inference on video file and MIPI CSI or Webcams.

* SSD MobileNet V2
* Segnet

## Know your camera
To get the I/O information or location of all the cameras connected, run the command ```v4l2-ctl --list-devices```
