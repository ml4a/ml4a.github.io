<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<title>{{ page.title }}</title>
	{% if page.includes contains 'mathjax' %}
		<link rel="stylesheet" type="text/css" href="/css/main.css">
		<link rel="icon" href="/images/favicon.png">
		<script type="text/x-mathjax-config">
		MathJax.Hub.Config({
  			CommonHTML: {scale: 100},
  			jax: ["input/TeX","output/HTML-CSS"],
  			tex2jax: {inlineMath: [["$","$"],["\\(","\\)"]]}
		});
		</script>
		<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
		</script>
	{% endif %}
	{% if page.includes contains 'jquery' %}
		<script src="/demos/libraries/jquery-1.8.3.min.js"></script>
	{% endif %}
	{% if page.includes contains 'convnetjs' %}
		<script src="/demos/libraries/convnet.js" type="text/javascript"></script>
		<script src="/demos/libraries/util.js" type="text/javascript"></script>
	{% endif %}
	{% if page.includes contains 'dataset' %}
		<script src="/demos/src/dataset.js" type="text/javascript"></script>
	{% endif %}
	{% if page.includes contains 'convnet' %}
		<script src="/demos/src/convnet.js" type="text/javascript"></script>
	{% endif %}
	{% if page.includes contains 'visualizer' %}
		<script src="/demos/src/visualizer.js" type="text/javascript"></script>
	{% endif %}
	</head>

	<body>

		<!-- not ready yet -->
		<!--{% assign quote = site.data.quotes.lovelace %}-->
		<!--{% include header.html quote=quote image_path=page.header_image %}--> 

		<div class="navbar">
			<nav>
	    		<ul>
					<li style="display:none;"><a href="#end-nav" class="skip-navigation">Skip Navigation</a></li>
	        		<li><a href="/index/">ml4a</a></li>
		        	<li><a href="/guides/">guides</a></li>
		        	<li><a href="/demos/">demos</a></li>
		        	<li><a href="/classes/">classes</a></li>
		        	<li><a href="https://github.com/ml4a">code</a></li>
		        	<li><a href="https://www.twitter.com/ml4a_">@</a></li>
	    		</ul>
			</nav>
		</div>


<style>

</style>
		<span id="end-nav"></span>
		<div class="container">
			{{ content }}
		</div>
		
		<footer>
    		<ul>
        		<li><a href="/about/">about</a></li>
        		<li><a href="/guides/Contribute/">contribute</a></li>
        		<li><a href="https://github.com/ml4a">github.com/ml4a</a></li>
			</ul>
		</footer>
<!--
		<script>
		  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
		  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
		  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
		  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

		  ga('create', 'UA-90023713-1', 'auto');
		  ga('send', 'pageview');
		</script>
-->
</script>
	</body>
</html>