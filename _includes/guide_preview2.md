<!-- {% assign page = site.works | where:"title", include.title | first %} -->

{% assign name = include.name %}
{% assign guide = site.data.guides[name] %}
{% assign title = guide.title %}
{% assign description = guide.description %}
{% assign summary = guide.summary %}
{% assign rootlink = "https://github.com/ml4a/ml4a-guides/tree/master/" %}
{% assign link = guide.link %}
{% assign thumb = guide.thumb %}


<div class="project">
	<a href="{{rootlink}}{{link}}">
		<img src="{{thumb}}">
		<div class="overlay">
			<div class="overlay_title">
				{{title}}
			</div>
<!-- 			<div class="overlay_description">
				{{description}}
			</div> -->
			<div class="overlay_summary">
				<ul>
				{% for s in summary %}
					<li>{{ s }}</li> 
				{% endfor %}			
				</ul>
			</div>
		</div>
	</a>
</div>