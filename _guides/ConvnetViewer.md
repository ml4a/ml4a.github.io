---
layout: guide
title: "Convnet Viewer"
---

ConvnetViewer is an [openFrameworks](http://www.openframeworks.cc) application, part of the [ml4a-ofx collection](https://github.com/ml4a/ml4a-ofx/), which lets you view the layer-by-layer activations of a trained convolutional neural network.

{% include figure_multi.md path1="/images/guides/convnetviewer.jpg" caption1="Activation maps for trained convolutional network analyzing an image of a dog." %}

The app can be used in either webcam mode or you can load an image instead. You may toggle through each of the layers to see its activation maps as the image propagates forward through the network. 

It has no additional functionality besides letting you see how a convnet works. For interesting use cases and applications, you may want to look at [ConvnetClassifier](/guides/ConvnetClassifier/), [ConvnetRegressor](/guides/ConvnetRegressor/), or [ConvnetOSC](/guides/ConvnetOSC/).

ConvnetViewer can be used to gain insights into how convnets work. It was used for this purpose in this short video, made originally for [Google's AI experiments](https://aiexperiments.withgoogle.com/what-neural-nets-see).

<center>
<iframe width="840" height="472" src="https://www.youtube.com/embed/Gu0MkmynWkw" frameborder="0" allowfullscreen></iframe>
</center>