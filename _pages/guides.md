---
layout: default
title: Guides
redirect_from: /guides2/
---

<style>
.project {
    width:280px;
    height:200px;
    margin:10px;
    padding:0px;
    position:relative;
    display:inline-block;
    text-align:left;
}

.overlay {
	width:100%;
    height:100%;
    position:absolute;
    top:0;
    left:0;
    display:inline-block;
    -webkit-box-sizing:border-box;
    -moz-box-sizing:border-box;
    box-sizing:border-box;
    color:white;
}

.overlay_title {
    font-size:1.25em;
    background:rgba(0,0,0,0.7);
    padding:7px;
}
/*
.overlay_description {
    font-size:1.1em;
    background:rgba(0,0,0,0.7);
    margin-top:0px;
    padding:4px;
    width: 100%;
    border-top: 1px solid rgba(255,255,255,0.45);
}*/
.overlay_summary {
    font-size:0.95em;
    background:rgba(0,0,0,0.7);
    display: none;
    margin-top:8px;
    /*padding:10px;*/
    width: 100%;
}
.project a:hover .overlay_summary {
    display:inline-block;
}
.overlay .overlay_summary li {
    padding:2px;
}



#platforms {
	margin-top:10px;
	margin-bottom:20px;
}
.platform {
	border: 1px solid #aaa;
	padding-bottom: 8px;
	padding-top: 8px;
	padding-left: 24px;
	padding-right: 24px;
	margin: 2px;
	display:inline-block;
}

</style>



<div id="platforms">
	<div id="platform_all" class="platform"><a href="javascript:displayAll();">All</a></div>
	<div id="platform_python" class="platform"><a href="javascript:displayByKey('python');">Keras / Tensorflow</a></div>
	<div id="platform_openframeworks" class="platform"><a href="javascript:displayByKey('openframeworks');">openFrameworks</a></div>
</div>


{% include guide_preview.md name="Contribute" %}

{% include guide_preview.md name="fundamentals" %}
{% include guide_preview.md name="intro-python" %}
{% include guide_preview.md name="intro-numpy" %}
{% include guide_preview.md name="kNN" %}
{% include guide_preview.md name="linear-regression" %}
{% include guide_preview.md name="diy-net" %}
{% include guide_preview.md name="simple" %}
{% include guide_preview.md name="keras-classification" %}
{% include guide_preview.md name="cnn" %}
{% include guide_preview.md name="transfer-learning" %}
{% include guide_preview.md name="quickdraw" %}
{% include guide_preview.md name="rnn" %}
{% include guide_preview.md name="seq2seq" %}
{% include guide_preview.md name="image-search" %}
{% include guide_preview.md name="image-path" %}
{% include guide_preview.md name="image-tsne" %}
{% include guide_preview.md name="audio-tsne" %}
{% include guide_preview.md name="text-retrieval" %}
{% include guide_preview.md name="wiki-tsne" %}
{% include guide_preview.md name="neural-painter" %}
{% include guide_preview.md name="word2vec" %}
{% include guide_preview.md name="eigenfaces" %}
{% include guide_preview.md name="autoencoders" %}
{% include guide_preview.md name="neural-synth" %}
{% include guide_preview.md name="qlearn" %}
{% include guide_preview.md name="qnets" %}
{% include guide_preview.md name="biggan" %}
{% include guide_preview.md name="biggan2" %}
{% include guide_preview.md name="glow" %}
{% include guide_preview.md name="maskrcnn" %}

{% include guide_preview.md name="AudioClassifier" %}
{% include guide_preview.md name="AudioTSNEViewer" %}
{% include guide_preview.md name="ConvnetOSC" %}
{% include guide_preview.md name="ConvnetClassifier" %}
{% include guide_preview.md name="ConvnetRegressor" %}
{% include guide_preview.md name="ConvnetViewer" %}
{% include guide_preview.md name="DoodleClassifier" %}
{% include guide_preview.md name="FaceClassifier" %}
{% include guide_preview.md name="FaceRegressor" %}

{% include guide_preview.md name="Gobot" %}
{% include guide_preview.md name="ImageTSNEViewer" %}
{% include guide_preview.md name="ImageTSNELive" %}
{% include guide_preview.md name="Pix2Pix" %}
{% include guide_preview.md name="ReverseImageSearchFast" %}
{% include guide_preview.md name="ReverseObjectSearchFast" %}
{% include guide_preview.md name="YoloLive" %}






<script>
// include guide_preview.md name="FacePredictor"
// include guide_preview.md name="ConvnetPredictor"

function highlightButton(keyword){
	document.getElementById("platform_python").style.border = "none";
	document.getElementById("platform_openframeworks").style.border = "none";
	document.getElementById("platform_all").style.border = "none";
	document.getElementById("platform_"+keyword).style.border = "1px solid #1abc9c";
}
function displayAll() {
	var d = document.getElementsByClassName("project");
	for(var i = 0; i < d.length; i++){ d[i].style.display = "inline-block"; }
	highlightButton('all');
};
function hideAll() {
	var d = document.getElementsByClassName("project");
	for(var i = 0; i < d.length; i++){ d[i].style.display = "none"; }	
};
function displayByKey(keyword) {
	hideAll();
	d = document.getElementsByClassName("project "+keyword);
	for(var i = 0; i < d.length; i++){ d[i].style.display = "inline-block"; }
	highlightButton(keyword);
};
displayAll();

</script>