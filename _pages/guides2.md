---
layout: default
title: Guides
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
	<div class="platform">Keras</div>
	<div class="platform">openFrameworks</div>
	<div class="platform">Processing</div>
	<div class="platform">p5.js</div>
</div>


{% include guide_preview2.md name="fundamentals" %}
{% include guide_preview2.md name="simple" %}
{% include guide_preview2.md name="cnn" %}
{% include guide_preview2.md name="transfer-learning" %}
{% include guide_preview2.md name="rnn" %}
{% include guide_preview2.md name="seq2seq" %}
{% include guide_preview2.md name="image-search" %}
{% include guide_preview2.md name="image-path" %}
{% include guide_preview2.md name="image-tsne" %}
{% include guide_preview2.md name="audio-tsne" %}
{% include guide_preview2.md name="text-retrieval" %}
{% include guide_preview2.md name="neural-painter" %}
{% include guide_preview2.md name="word2vec" %}
{% include guide_preview2.md name="qlearn" %}
{% include guide_preview2.md name="qnets" %}





{% include guide_preview2.md name="AudioTSNEViewer" %}
{% include guide_preview2.md name="ConvnetClassifier" %}
{% include guide_preview2.md name="DoodleClassifier" %}
{% include guide_preview2.md name="FaceClassifier" %}
{% include guide_preview2.md name="FaceRegressor" %}
{% include guide_preview2.md name="ImageTSNEViewer" %}
{% include guide_preview2.md name="ImageTSNELive" %}
{% include guide_preview2.md name="Pix2Pix" %}
{% include guide_preview2.md name="ReverseImageSearchFast" %}
{% include guide_preview2.md name="ReverseObjectSearchFast" %}
{% include guide_preview2.md name="YoloLive" %}


{% include guide_preview2.md name="Contribute" %}

