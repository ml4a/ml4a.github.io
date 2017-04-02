---
layout: guide
title: "FaceRegressor"
---

FaceRegressor is an [openFrameworks](http://www.openframeworks.cc) application, part of the [ml4a-ofx collection](https://github.com/ml4a/ml4a-ofx/), which rapidly lets you train a regression model and control simple graphics by doing facial expressions. 

FaceRegressor is based on [ofxFaceTracker2](https://github.com/HalfdanJ/ofxFaceTracker2) and [ofxGrt](https://github.com/nickgillian/ofxGrt). It has been used by [Andreas Refsgaard](https://andreasrefsgaard.dk/) as part of integrating machine learning into the [Eye Conductor](https://andreasrefsgaard.dk/project/eye-conductor/) project. 

{:.center}
![faceRegressor gif](http://andreasrefsgaard.dk/wp-content/uploads/2016/12/FaceRegression.gif)


## Applications

The application in itself is very minimal, but can be expanded to control parameters in other applications like Ableton Live:

<center>
<iframe src="https://player.vimeo.com/video/197499111" width="640" height="360" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
</center>




## Training instructions

1. Set the height and width of the rectangle using [+], [-], [1] and [2] keys.

2. Press [r] to record some training samples containing your selected facial features (gestures, orientation and/or raw points) and the height and width of the rectangle.

3. Repeat step 1) and 2) with different rectangle width and height and different facial expressions / head orientations

4. Press [t] to train the model.

5. See the changes in rectangle width and height based on your facial orientation and expression.


The video below takes you through the steps of the training process.

<center>
<iframe src="https://player.vimeo.com/video/197501274" width="640" height="360" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
</center>





