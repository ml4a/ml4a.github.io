---
layout: guide
title: "FaceRegressor"
---

FaceRegressor is an [openFrameworks](http://www.openframeworks.cc) application, part of the [ml4a-ofx collection](https://github.com/ml4a/ml4a-ofx/), which rapidly lets you train a regression model and control simple graphics by doing facial expressions. 

FaceRegressor is based on [ofxFaceTracker2](https://github.com/HalfdanJ/ofxFaceTracker2) and [ofxGrt](https://github.com/nickgillian/ofxGrt). It has been used by [Andreas Refsgaard](https://andreasrefsgaard.dk/) as part of integrating machine learning into the [Eye Conductor](https://andreasrefsgaard.dk/project/eye-conductor/) project. 

{:.center}
![faceRegressor gif](http://andreasrefsgaard.dk/wp-content/uploads/2017/04/FaceRegressor.gif)


## Applications

The application in itself is very minimal, but can be expanded to control parameters in other applications like Ableton Live:

<center>
<iframe src="https://player.vimeo.com/video/197499111" width="640" height="360" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
</center>




## Training instructions

1. Set the height and width of the rectangle using the sliders

2. Click record and prepare a distinct facial expression. Once recording, your training examples will contain containing your selected facial features (gestures, orientation and/or raw points) and the current width and height of the rectangle.

3. Repeat step 1-2 with different width and height values for the rectangle and different expressions

4. Click train to train the model

5. Move your face and see the changes in rectangle width and height


The video below takes you through the steps of the training process.

<center>
<iframe src="https://player.vimeo.com/video/212934622" width="640" height="360" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
</center>





