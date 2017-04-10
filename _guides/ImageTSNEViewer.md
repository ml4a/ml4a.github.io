---
layout: guide
title: "Image t-SNE viewer"
---

This app embeds a set of image files in 2d using using the [t-SNE dimensionality reduction technique](https://lvdmaaten.github.io/tsne/), placing images of similar content near each other, and lets you browse them with the mouse and scroll wheel.

An example of a t-SNE of images may look like the below figure. Because there is a lot of content in a figure containing so many images, we are sometimes interested in browsing it with more dexterity; as in being able to zoom, scroll, rescale, and so on.

{% include figure.html path="/images/guides/image-tsne-highlights.jpg" caption="t-SNE of images of animals" %}

## Installation
---

For the analysis portion, you need the following python libraries installed: [scikit-learn](http://scikit-learn.org/stable/install.html), [keras](https://keras.io/), [numpy](https://docs.scipy.org/doc/numpy/user/install.html), and [simplejson](https://simplejson.readthedocs.io/en/latest/).

The openFrameworks application only requires one addon: [ofxJSON](https://github.com/jeffcrouse/ofxJSON).


## Run the analysis
---

The analysis uses [keras](https://keras.io/) and analyzes each image in your dataset by extracting the "fc2" (last pre-classification layer) activations from a trained 

First prepare a folder of images to analyze. The images may be distributed into subfolders to any depth, since the search through the root folder is recursive.

In a terminal or command prompt, run the script `tSNE-images.py` (found in the `apps` folder of `ml4a-ofx`) to generate the t-SNE layout. Specify a directory of images and some parameters:


	python tSNE-images.py --images_path path/to/input/directory --output_path path/to/output/json

for example:

	python tSNE-images.py --images_path ../datasets/animals/ --output_path ../apps/ImageTSNEViewer/bin/data/points.json

After doing the analysis, the script will save a json file with the layout t-SNE coordinates at the path specified.

#### Optional parameters

// alpha, 


## Run the viewer application
---

If you are building the application from source, just make sure the variable `path` is set to point to the JSON file. If you are running the pre-compiled application, you need to rename the JSON file to `points.json` and place it in the app's `data` folder, which is the default file path it is opening.

You should get something that looks like this.

![Audio t-SNE](/images/guides/image-tsne.jpg)


## Run the viewer
--
The `ofApp` must have the correct path to the JSON file. By default, it will look in the `data` folder, or you may change the path to the variable `____` and recompile the app.

[IMAGE OF VIEWER]

The application lets you move around the t-SNE by dragging the mouse around. You may also zoom in and out by scrolling the mouse.

The gui also has some parameters

