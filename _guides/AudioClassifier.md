---
layout: guide
title: "AudioClassifier"
---
### What is it?

AudioClassifier is an [openFrameworks](http://www.openframeworks.cc) application, part of the [ml4a-ofx collection](https://github.com/ml4a/ml4a-ofx/), which can be trained to distinguish different sounds from each other and output the result to other applications over OSC. 

AudioClassifier has been used by [Støj](http://stoj.io) to play Wolfenstein 3D only using sounds. 

<center>
<iframe src="https://player.vimeo.com/video/207831279" width="640" height="360" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
</center>



## Training instructions

1. Press the record button to start/stop recording training examples on the current class.

2. Switch the class label by pulling the *Class Label* slider and record new training examples

3. Repeat step 2 for however many distinct sounds you intend to classify. 

4. Press Train to train the model

5. Make some sounds and see how they get classified

The video below takes you through the steps of the training process.

<center>
<iframe src="https://player.vimeo.com/video/212739123" width="640" height="360" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
</center>


## Threshold mode
When *Threshold Mode* is on, the app will only react to sounds louder (RMS) than *volThreshold*. 

Additionally the *Threshold timer (ms)* will block any new inputs for a number of milliseconds.

Threshold mode effects both when training new examples and classifing new inputs. 

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
* m: toggle threshold mode


## Setup and training considerations

* Try to keep background noise to a minimum.

* Use sounds that are easy to distinguish from each other to achieve  higher levels of correct classifications. 

* If the classification is not accurate enough, try to record more training examples


