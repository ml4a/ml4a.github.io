---
layout: guide
title: "YOLO Live"
---

This application runs real-time multiple object detection on a video input. 

YOLO stands for "you only look once," referring to the way the object detection is implemented, where the network is restricted to determine all the objects along with their confidences and bounding boxes, in one forward pass of the network for maximum speed. It is based on the [paper](https://arxiv.org/abs/1612.08242) and associated library [Darknet](https://pjreddie.com/darknet/yolo/).

YOLO can take as its input a webcam, video file, or screengrabber. The screengrabber lets you detect objects inside anything you can put on your screen, such as internet video streams. Some example screenshots below:

{% include figure_multi.md path1="/images/guides/yolo-screen2.png" caption1="YOLO can be used on Facebook Live and other real-time streams" path2="/images/guides/yolo-screen3.png" caption2="Detecting objects in YouTube" path3="/images/guides/yolo-screen1.png" caption3="What's in San Diego zoo cam?" %}


## Installation
---

This addon requires [ofxDarknet](https://github.com/mrzl/ofxDarknet) to be compiled with [CUDA](https://developer.nvidia.com/cuda-downloads) support. As such, this application will not currently work on any machine which does not have an NVIDIA graphics card and CUDA properly installed. Directions for installing ofxDarknet can be found on [the GitHub readme](https://github.com/mrzl/ofxDarknet).

Additionally, the app requires [ofxScreenGrab](https://github.com/genekogan/ofxScreenGrab) for using the screen pixels as an input source.

## Usage instructions
---

YOLOLive has 3 options for input -- webcam, video file, and screengrabber. The screengrabber allows you to change the picker window if you toggle on the "set window" option and drag the top-left and bottom-right corners to get the window you want.

For the screengrabber, another button on the GUI, "MacBook retina?" must be set on if you have the retina display activated (note: retina might get turned off if you are connected to an external monitor or projector).

Another slider, `threshold` controls the minimum confidence threshold to include detected objects. Objects whose probability is above threshold get drawn to the screen, so the lower the threshold, the more detected objects you will get.