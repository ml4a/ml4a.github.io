---
layout: guide
title: "Reverse Image Search (Fast)"
---

This app demonstrates how to quickly retrieve the k-nearest neighbors to an image query. It is the same concept as what's given in the [Reverse Image Search](/guides/ReverseImageSearch/), except that it can make queries on previously-unseen images at nearly real-time, whereas the former is only a viewer for a fixed set of images pre-analyzed by a Python script. It is also mostly identical to [Reverse Object Search Fast](/guides/ReverseObjectSearchFast/), except that it operates over the entire image rather than over detected objects within the image.

{% include figure_multi.md path1="/images/guides/RIS_tennis.jpg" caption1="Live reverse image search at ~1 fps" %}


## How it works
---

A large dataset of images is pre-analyzed by a convolutional neural network trained to classify images. A feature vector is extracted from each image, the 4096-bit activations from the last fully-connected layer (just before the classification layer), and recorded, along with the image's original file path.

After all feature vectors are extracted, they are transformed into a lower dimensional representation (for dimensionality reduction) using a [random matrix projection](http://stats.stackexchange.com/questions/235632/pca-vs-random-projection). It can optionally be done with principal component analysis instead, which is probably a bit more accurate, but runtime can be very high for a large corpus.

After feature extraction and dimensionality reduction, the results are serialized into a `.dat` file so you can load them back later.  

After the analysis has been done or a previous analysis has been loaded, the reduced feature vectors are embedded into a [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) for fast lookup.  

At runtime, a new image is queried by being analyzed in exactly the same procedure as above -- its feature vector, extracted from the same layer as above is projected onto the lower-dimensional subspace using the random matrix generated during the analysis phase. The projected vectors are each used to query the KD-Tree which retrieves the approximate nearest `k` neighbors to each object respectively (`k` is set using the `num results` slider). Those images are then loaded from disk and displayed.

## Installation
---
The following addons are needed:

- [ofxCcv](https://github.com/kylemcdonald/ofxCcv)
- [ofxKDTree](https://github.com/genekogan/ofxKDTree)
- [ofxLearn](https://github.com/genekogan/ofxLearn)

Make sure the application finds the correct path to the network weights file, `image-net-2012.sqlite3`. 

## Usage instructions
---

### Analyzing directory of images

In order to be able to retrieve nearest neighbor images, a dataset of images needs to first be analyzed. Through the interface, this can be done by clicking `analyze directory`, which will compile the image paths (recursively) from a base directory. Equivalently, run the following code:


	baseDir = "YOUR_IMAGES_DIRECTORY_HERE";
    extractFeaturesForDirectory(toExtractDir);
    runDimReduction();
    runKDTree();

The analysis may take hours, depending on how many images are in the directory. Each image can take around 0.25-0.75 sec depending on your machine.  

Once analysis is done, it can be saved by clicking `save` or running:

    save("YOUR_VECTORS_FILE.dat");

If you want to save just the original vectors without the projections (useful to save them while doing the analysis, in case it crashes in the middle...) you can set the `featuresOnly` bool to true and run:

	save("YOUR_VECTORS_FILE.dat", true);
	
To load the saved vectors back into memory next time you launch the application, click `load vectors` and select the saved `dat` file, then select the base directory where the images are located. Equivalently, this can be done in code:

	baseDir = "/PATH_TO_IMAGE_DIRECTORY";
    load("/PATH_TO_SAVED_VECTORS", baseDir);

### Download pre-analyzed image set

An example image set was analyzed and prepared, which you can download and use. The image set is [MC-Coco](http://mscoco.org/), and the saved analysis contains vectors from around 145,000 unique images.  

First you need to [download](http://mscoco.org/dataset/#download) the images. Download the files (around 25GB altogether):

- 2014 Training images [80K/13GB]
- 2014 Val. images [40K/6.2GB]
- 2014 Testing images [40K/6.2GB]

Unzip them and place the images into side-by-side folders 'train', 'test', and 'val' respectively. Make sure the images are unzipped this way because the trained vectors are saved for these relative paths. For example, the first image in the the `test` directory should be called something like `/PATH_TO_COCO/test/COCO_test2014_000000000014.jpg` where `/PATH_TO_COCO/` is the location of the folder containing the images.

Next, download the saved vectors. It can be downloaded from Dropbox via the following link:

[https://drive.google.com/drive/folders/0B3WXSfqxKDkFRE05MXY1U3c0YVU](https://drive.google.com/drive/folders/0B3WXSfqxKDkFRE05MXY1U3c0YVU)
	
Afterwards, you may load them as usual following the method in the previous section. Make sure `baseDir` is pointing towards the parent folder of `train`, `test`, and `val`.

### Runtime

Once vectors are loaded, the application can analyze a new image and find nearest neighbors for each detected object. There are four input options, which are selected via toggle.

1) `query random image`: this will randomly load an image from the analyzed images (in `baseDir`) and do the analysis on it.

2) `query webcam`: this will turn on your webcam and begin to analyze it continuously.

3) `query video`: this will let you load a video file from disk and begin playing it and analyzing it continuously.

4) `query screengrab`: this will let you load your screen pixels below the application window as the query image so you can do real-time reverse image search on your desktop, browser, etc. If you are on Mac, make sure you set the `retina` variable accordingly. You can toggle `set screengrab window` to set the position and size of the grabbr.

There are two display parameters to set also:

- `header height` and `thumb height` simply control the display height of the query image and the resulting image rows. 
- `num results` controls the number of nearest neighbors to load for each detected object.
