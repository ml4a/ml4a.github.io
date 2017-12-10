---
layout: guide
title: "FaceClassifier"
---

FaceClassifier is an [openFrameworks](http://www.openframeworks.cc) application, part of the [ml4a-ofx collection](https://github.com/ml4a/ml4a-ofx/), which rapidly lets you train a classifier to recognize facial expressions. 

FaceClassifier is based on [ofxFaceTracker2](https://github.com/HalfdanJ/ofxFaceTracker2) and [ofxGrt](https://github.com/nickgillian/ofxGrt). It has been used by [Andreas Refsgaard](https://andreasrefsgaard.dk/) as part of integrating machine learning into the [Eye Conductor](https://andreasrefsgaard.dk/project/eye-conductor/) project. 

{:.center}
![faceClassifier gif](http://andreasrefsgaard.dk/wp-content/uploads/2017/04/FaceClassifier.gif)

## OSC Output
By default the app outputs OSC to localhost, port 8000, adress "/classification". This can be changed in ofApp.h.


## Key inputs
As an alternative to using the GUI you can use the following key inputs:

* 1-9: set the class label
* s: save model
* l: load model
* t: train classifier
* c: clear examples
* r: toggle recording
* p: pause prediction

## Training instructions

1. To record training examples for class 1 make a distinct facial expression and click [record].

2. Set the Class Label slider to 2 and click [record] to record examples of a different facial expression.

3. Repeat this for class 3 or however many distinct expressions you want to classify. Remember to change the classes by pressing the numeric keys accordingly. 

4. Click [train] to train the model.

5. Make facial expressions and see what class the model predicts.


The video below takes you through the steps of the training process.

<center>
<iframe src="https://player.vimeo.com/video/213068540" width="640" height="360" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
</center>
