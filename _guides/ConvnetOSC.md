---
layout: guide
title: "ConvnetOSC"
---

### What is it?

ConvnetOSC is an [openFrameworks](http://www.openframeworks.cc) application, part of the [ml4a-ofx collection](https://github.com/ml4a/ml4a-ofx/), which sends ConvNet activations as a 4096-bit input vector over OSC. Used in combination with [Wekinator](http://www.wekinator.org/) ConvnetOSC lets you train a classifier to recognise whatever objects, persons, postures, drawings and other visuals you show it through your webcam. It can also be used for regression. 

{:.center}
![ConvnetOSC gif](https://andreasrefsgaard.dk/wp-content/uploads/2017/04/ConvnetOSC.gif)


ConvnetOSC has been used by [Andreas Refsgaard](http://andreasrefsgaard.dk/) in his installation [IS IT FUNKY?](https://andreasrefsgaard.dk/project/is-it-funky/) which tries distinguish funky images from boring ones after being trained on 15.000 (highly subjective) images from Google searches.

<center>
<iframe src="https://player.vimeo.com/video/197020660" width="640" height="360" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
</center>



## Training instructions

The video below takes you through the steps of the training process.

<center>
<iframe src="https://player.vimeo.com/video/212722917" width="640" height="360" frameborder="0	" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
</center>

To change the OSC host, port, and address, edit the settings.xml file in the data folder. By default, it is set to localhost, port 6448, address "/wek/inputs", which works out of the box with Wekinator.



## Setup and training considerations
ConvnetOSC is a very versatile application that can be used in a wide variety of setups. However, to optimise correct classification/regression, some simple rules of thumbs are:

* Ensure variety in your training data: Say you want to classify whether someone is holding an object in their hands, try to record training samples that shows the person holding the object in different postures and positions, close to the camera, far from the camera, in both hands etc. 

* Keep your settings stable when using small training samples: Training in different sets of lighting conditions and hereafter attempting to classify in yet another environment might prove difficult. The same is true for camera positions and angles, backgrounds, etc. Unless you intentionally want a classifier that is can handle different settings (which would require more training examples), you can save yourself a lot of sweat by keeping your physical setup stable. 

* If the classification is not accurate enough, try to record more training examples. Alternatively you can try different classifier algorithms inside Wekinator.

## How does a convnet work?


{::nomarkdown}
<iframe width="560" height="315" src="https://www.youtube.com/embed/Gu0MkmynWkw" frameborder="0" allowfullscreen></iframe>
{:/nomarkdown}