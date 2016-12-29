---
layout: guide
title: "DoodleClassifier"
---

### What is it?

DoodleClassifier is an [openFrameworks](http://www.openframeworks.cc) application, part of the [ml4a-ofx collection](https://github.com/ml4a/ml4a-ofx/), which lets you train a classifier to accurately recognize drawings ("doodles") from a camera.

It was first used in a project called [DoodleTunes](https://vimeo.com/197026662) by [Andreas Refsgaard](https://andreasrefsgaard.dk/) and [Gene Kogan](https://www.genekogan.com/), which used the app to recognize doodles of musical instruments and turn them into music being made in Ableton Live. It was inspired by the [QuickDraw app](https://quickdraw.withgoogle.com/) made by [Jonas Jongejan, Henry Rowley, and collaborators at Google Creative Lab](https://www.youtube.com/watch?v=X8v1GWzZYJ4).

<center>
<iframe src="https://player.vimeo.com/video/197026662" width="800" height="450" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
</center>

## Physical setup

DoodleClassifier is quick and easy to set up. It often works best to set up an overhead camera pointing downward at your drawing surface, but in principle can work anywhere that the camera can be pointed at. It's also advised to use thick markers rather than pencils or pens, because the lines can be more easily distinguished by the software. The picture below shows the setup that was used for DoodleTunes, where the camera is positioned at the top of a stand, facing downward at the paper below.

{:.center}
![DoodleClassifier setup](/images/guides/doodleclassifier_setup.jpg "DoodleClassifier setup")

## Settings

Before launching the app, review and adjust the settings, which can be found in the file `settings_doodleclassifier.xml`. In that file, you must define the classes you'd like for the app to recognize. By default, they are `circle`, `star`, and `arrow`, but you may change those and have as many different classes as you'd like.

Once the app is trained, classifications are sent over OSC (open sound control), to the IP address and port specified in the settings file, `localhost:5000` by default, using the address `\classification`. You may change these accordingly, and will easily send to another computer over the same network if you change the IP address.

## Training

During training, you are giving the app as many examples of each class as you can. The first slider at the top of the GUI lets you select one of your defined classes as the active class. On a piece of paper, draw some instances of that class and put them underneath the camera. You will probably need to adust the computer vision settings to properly identify and segment them. A screenshot from drawing some instances of circles, and an explanation of what the CV parameters do is given below.

{:.center}
![DoodleClassifier interface](/images/guides/doodleclassifier_interface.jpg "DoodleClassifier interface")

The CV parameters are found at the bottom of the GUI. The first slider you should adjust is the `Threshold` which determines the brightness threshold by which to separate the foreground from background content. Here you may have to adjust your physical setup to minimize the influence of shadows, which may interfere with this process. An overhead light may be helpful.

The `Dilations` slider dilates (thickens) the discovered lines, and may help reduce fragmentation of found instances.

Finally, the `Min area` and `Max area` sliders control the acceptable range of sizes of drawn instances to allow. If you set `Min area` too low, you may have a lot of spurious doodles discovered. The ideal is to have segmented your doodles (inside the green rectangles), but not anything else, which may corrupt your classifier. 

The screenshot above is an example of successful segmentation. When you have achieved this with your first class, click `Add samples`, and after a moment, during which a [convnet analyzes the features of each doodle](/ml4a/convnets/) and saves the feature vector to memory, the samples will appear below the camera images. For example: 

{:.center}
![DoodleClassifier circles](/images/guides/doodleclassifier_class1.jpg "DoodleClassifier circles")

Now repeat this process with all of your classes, by moving the first slider in the GUI to each class in order, and drawing instances of that class underneath the camera, and clicking `Add samples`. For example, for stars and arrows, you should see something like:

{:.center}
![DoodleClassifier stars](/images/guides/doodleclassifier_class2.jpg "DoodleClassifier stars")

{:.center}
![DoodleClassifier arrows](/images/guides/doodleclassifier_class3.jpg "DoodleClassifier arrows")

How many intances of each class do you need to train accurately? There is no general way to answer this, as this depends heavily on the nature and quality of your image classes, and may vary from just a handful to several hundreds of images. Some intuitions are helpful; the following things make the classification task more complicated and therefore require more training samples:

- the more distinct classes you have
- the internal variance of each class; are the drawings within each class relatively homogenous or do they greatly vary?
- the similarity between pairs of image classes you've defined -- the more similar two classes are to each other, the more likely your classifier will demonstrate some confusion between them.

In this simple toy example within this tutorial, we were able to get away with providing just 5-10 samples of each image to get decent accuracy, but usually this won't be enough. For [DoodleTunes](https://vimeo.com/197026662), roughly 30-50 samples were drawn for each class, in order to reduce error due to large variations in handwriting styles in multiple people drawing examples of each class. If you have a particularly ambitious task, you may need hundreds of samples. Experimentation should help determine the right numbers for your application.

Once you are ready, click `Train` in the interface, and wait for the training to complete. This may take anywhere from a few seconds to a few minutes depending on the complexity of your dataset.

## Prediction

Once training has completed, you can classify new images. Draw some instances of your class, put them under the camera, and click `Classify`. After a moment, it will segment your doodles as before, and predict which classes they belong to. It will instantly send each predicted class as an OSC message to the address given in the settings, where the value of the OSC message is a string corresponding to the name of the predicted class. See below for an example.

{:.center}
![DoodleClassifier prediction](/images/guides/doodleclassifier_prediction.jpg "DoodleClassifier prediction")

## Summary

The video below shows this whole process. Once you get used to the process, it should only take you a few minutes to setup.

<center>
<iframe src="https://player.vimeo.com/video/196944929?color=1abc9c" width="800" height="500" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
</center>
