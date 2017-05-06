---
layout: guide
title: "YOLO Live"
---

This application runs real-time multiple object detection on a video input. 

YOLO stands for "you only look once," referring to the way the object detection is implemented, where the network is restricted to determine all the objects along with their confidences and bounding boxes, in one forward pass of the network for maximum speed. 


## Installation
---

This addon requires [ofxDarknet](___) to be compiled with [CUDA](__) support. As such, this application will not currently work on any machine which does not have an NVIDIA graphics card and CUDA properly installed. Directions for installing ofxDarknet can be found on [the GitHub readme](___).

Additionally, the app requires [ofxScreenGrab]() for using the screen pixels as an input source, although this may be removed if you don't intend to use that an input source.


yolo-screen1.png
yolo-screen2.png
yolo-screen3.png