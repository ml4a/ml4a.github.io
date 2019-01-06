---
layout: default
title: Demos
redirect_from: /demos2/
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

/*.overlay_description {
    font-size:1.1em;
    background:rgba(0,0,0,0.7);
    margin-top:0px;
    padding:4px;
    width: 100%;
    border-top: 1px solid rgba(255,255,255,0.45);
}*/
.overlay_summary {
    font-size:1.1em;
    background:rgba(0,0,0,0.7);
    display: none;
    margin-top:8px;
    padding:10px;
    width: 90%;
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
	<div id="platform_ml5" class="platform"><a href="javascript:displayByKey('ml5');">ml5.js</a></div>
	<div id="platform_demo" class="platform"><a href="javascript:displayByKey('demo');">Demos</a></div>
	<div id="platform_figure" class="platform"><a href="javascript:displayByKey('figure');">Figures</a></div>
</div>

{% include demo_preview.md name="Contribute" %}


{% include demo_preview.md name="ml5_classifier" %}
{% include demo_preview.md name="ml5_image" %}
{% include demo_preview.md name="ml5_sound" %}
{% include demo_preview.md name="ml5_camera" %}
{% include demo_preview.md name="ml5_guitar" %}
{% include demo_preview.md name="ml5_mobilenet" %}
{% include demo_preview.md name="ml5_speech" %}
{% include demo_preview.md name="ml5_regression" %}
{% include demo_preview.md name="ml5_regression_generative" %}
{% include demo_preview.md name="ml5_regression_pong" %}
{% include demo_preview.md name="ml5_playback" %}
{% include demo_preview.md name="ml5_posenet" %}
{% include demo_preview.md name="ml5_posenet_sound" %}
{% include demo_preview.md name="ml5_posenet_nose" %}
<!--
% include demo_preview.md name="facetracker_knn" %
-->

{% include demo_preview.md name="tsne_viewer" %}
{% include demo_preview.md name="simple_forward_pass" %}
{% include demo_preview.md name="forward_pass_mnist" %}
{% include demo_preview.md name="forward_pass_cifar" %}
{% include demo_preview.md name="convolution" %}
{% include demo_preview.md name="convolution_all" %}
{% include demo_preview.md name="confusion_mnist" %}
{% include demo_preview.md name="confusion_cifar" %}
{% include demo_preview.md name="mnist_weights" %}
{% include demo_preview.md name="cifar_weights" %}

{% include demo_preview.md name="f_cifar_grid" %}
{% include demo_preview.md name="f_mnist_1layer" %}
{% include demo_preview.md name="f_mnist_grid" %}
{% include demo_preview.md name="f_mnist_input" %}
{% include demo_preview.md name="f_mnist_weights" %}
{% include demo_preview.md name="f_mnist_net" %}
{% include demo_preview.md name="f_neural_net" %}
{% include demo_preview.md name="f_neuron" %}
{% include demo_preview.md name="f_weights_analogy" %}

<script>

function highlightButton(keyword){
	document.getElementById("platform_figure").style.border = "none";
	document.getElementById("platform_ml5").style.border = "none";
	document.getElementById("platform_demo").style.border = "none";
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