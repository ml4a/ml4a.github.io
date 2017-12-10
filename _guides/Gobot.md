---
layout: guide
title: "Gobot"
---

[Code here](https://github.com/ml4a/ml4a-ofx/tree/master/apps/Gobot)

This application assists you in playing competitive Go online. It uses a [convolutional neural policy network](https://pjreddie.com/darknet/darkgo-go-in-darknet/) to "recommend" (i.e. predict) the next move a human player should make. The app uses a screengrabber and OpenCV template matching to scrape board position from your browser and automatically detect the current board position at any point in time. At this time, the CV detection is made specifically to detect pieces from [online-go.com](https://www.online-go.com). It can allegedly play at the level of a [1-dan](https://en.wikipedia.org/wiki/Go_ranks_and_ratings).

Below is a short snippet of a game played with my AI helper.

<center>
<iframe src="https://player.vimeo.com/video/221420426" width="800" height="488" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
</center>


## Installation

To build from source, you need the following addons.

- [ofxDarknet](https://github.com/mrzl/ofxDarknet)
- [ofxTemplateMatching](https://github.com/genekogan/ofxTemplateMatching)
- [ofxScreenGrab](https://github.com/genekogan/ofxScreenGrab)

Or the built application can be downloaded from [ml4a-ofx releases](https://github.com/ml4a/ml4a-ofx/releases).


## Instructions

After launching Gobot, you will see an empty Go board. At this point, you want to launch a new game on [online-go](https://www.online-go.com), and maximize the window.

You need to adjust the screengrabber window by pressing the 'd' key and then adjusting the corners of the screengrabber window to the top-left and bottom-right corners of the board in the browser, as seen below:

{% include figure_multi.md path1="/images/guides/gobot-screengrabber.jpg" caption1="Setting corners of window from which to grab Go board" %}

Then you can begin playing and making moves in the browser. At any given point, you can shift back to the application and it will give you a set of ordered recommendations, as in the following:

{% include figure_multi.md path1="/images/guides/go-recommends.jpg" caption1="Darknet recommendations for the next move" %}

Good luck!!