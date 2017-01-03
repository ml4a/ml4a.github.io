---
layout: guide
title: "FaceClassifier"
---

FaceClassifier is an [openFrameworks](http://www.openframeworks.cc) application, part of the [ml4a-ofx collection](https://github.com/ml4a/ml4a-ofx/), which rapidly lets you train a classifier to recognize facial expressions. 

FaceClassifier is based on [ofxFaceTracker2](https://github.com/HalfdanJ/ofxFaceTracker2) and [ofxGrt](https://github.com/nickgillian/ofxGrt). It has been used by [Andreas Refsgaard](https://andreasrefsgaard.dk/) as part of integrating machine learning into the [Eye Conductor](https://andreasrefsgaard.dk/project/eye-conductor/) project. 

{:.center}
![faceClassifier gif](http://andreasrefsgaard.dk/wp-content/uploads/2016/12/FaceClassifier.gif)

## Training instructions

1. To record training examples for class 1 make a distinct facial expression and press "r".

2. Press [2] to switch the class label and press [r] to record examples of a different facial expression.

3. Repeat this for class 3 or however many distinct expressions you want to classify. Remember to change the classes by pressing the numeric keys accordingly. 

4. Press [t] to train the model.

5. Make facial expression and see what class the model predicts.


The video below takes you through the steps of the training process.

<center>
<iframe src="https://player.vimeo.com/video/197503168" width="640" height="360" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
</center>





