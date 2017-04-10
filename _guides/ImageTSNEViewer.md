---
layout: guide
title: "Image t-SNE viewer"
---

[Code here](https://github.com/ml4a/ml4a-ofx/tree/master/apps/ImageTSNEViewer)

This app embeds a set of image files in 2d using using the [t-SNE dimensionality reduction technique](https://lvdmaaten.github.io/tsne/), placing images of similar content near each other, and lets you browse them with the mouse and scroll wheel.

An example of a t-SNE of images may look like the below figure. Because there is a lot of content in a figure containing so many images, we are sometimes interested in browsing it with more dexterity; as in being able to zoom, scroll, rescale, and so on.

{% include figure.html path="/images/guides/image-tsne-highlights.jpg" caption="t-SNE of images of animals from <a href=\"http://www.vision.caltech.edu/Image_Datasets/Caltech101/\">CalTech-101</a>" %}

## Installation

For the analysis portion, you need the following python libraries installed: [scikit-learn](http://scikit-learn.org/stable/install.html), [keras](https://keras.io/), [numpy](https://docs.scipy.org/doc/numpy/user/install.html), and [simplejson](https://simplejson.readthedocs.io/en/latest/).

The openFrameworks application only requires one addon: [ofxJSON](https://github.com/jeffcrouse/ofxJSON).

If you'd like to do the analysis and t-SNE directly in an openFrameworks app (without doing the analysis beforehand in Python), see the addon [ofxTSNE](https://github.com/genekogan/ofxTSNE). A guide for doing these t-SNEs live will be added in the future.

---

## Run the analysis

The analysis uses [keras](https://keras.io/) and analyzes each image in your dataset by extracting the "fc2" (last pre-classification fully-connected layer) activations from a trained convolutional neural network, that of [VGG-16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).

First prepare a folder of images to analyze. The images may be distributed into subfolders to any depth, since the search through the root folder is recursive.

There are two ways to run the analysis. One is to go through the [Python guide](https://github.com/ml4a/ml4a-guides/blob/master/notebooks/image-tsne.ipynb) and save the generated JSON at the end of the notebook. Alternatively, a convenient command-line tool `tSNE-images.py` is included in the `scripts` folder of `ml4a-ofx`. The first option is recommended to get a better understanding of how the analysis works, in which case you may shift there and come back with the JSON file and resume to the next section. 

If you are using the convenience script, the following instructions will handle the image feature extraction and t-SNE assignment. To run it on a directory of sounds, run the following command:

<code>
python tSNE-images.py --images_path path/to/input/directory --output_path path/to/output/json
</code>

This will analyze all the images in `path/to/input/directory` and assign a set of t-SNE coordinates to them saved in the file `path/to/output/json`.

for example:

<code>
python tSNE-images.py --images_path ../datasets/animals/ --output_path ../apps/ImageTSNEViewer/bin/data/points.json
</code>

You may optionally set the perplexity of the t-SNE using the `--perplexity` argument (defaults to 30), or the learning rate using `--learning_rate` (default 150). 

Note, you can also optionally change the number of dimensions for the t-SNE with the parameter `--num_dimensions` (defaults to 2) but this ofApp is currently setup to just read 2 columns at the moment.

After the analysis, you should have generated a JSON file containing the file paths to the individual images and their t-SNE embedding assignment. Make sure you do not move the images to another location after doing the analysis, because the paths are hardcoded into the JSON file.

---

## Run the viewer application

If you are building the application from source, just make sure the variable `path` is set to point to the JSON file. If you are running the pre-compiled application, you need to rename the JSON file to `points.json` and place it in the app's `data` folder, which is the default file path it is opening.

You should get something that looks like this.

{% include figure.html path="/images/guides/image-tsne.jpg" caption="t-SNE of images arranged by content similarity, using convnet fc2 activations" %}

The application lets you move around the t-SNE by dragging the mouse around. You may also zoom in and out by scrolling the mouse up and down, or adjusting the `scale` parameter.

The gui also has some parameters

`scale` : The total size (as a multiplier of screen width) of the canvas to display the images on.

`imageSize` : A scaling factor to resize the images. 

You may also click the button `save screenshot` which will save the entire t-SNE (not just what fits inside the display window) as a file `out.png` which you can find in the app's `data` folder.
