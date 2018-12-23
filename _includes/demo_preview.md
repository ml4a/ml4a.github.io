<!-- {% assign page = site.works | where:"title", include.title | first %} -->

{% assign name = include.name %}
{% assign demos = site.data.demos[name] %}
{% assign title = demos.title %}
{% assign description = demos.description %}
{% assign category = demos.category %}
{% assign link = demos.link %}
{% assign thumb = demos.thumb %}


<div class="project {{category}}">
	<a href="{{link}}">
		<img src="{{thumb}}">
		<div class="overlay">
			<div class="overlay_title">
				{{title}}
			</div>
<!-- 		<div class="overlay_description">
				{{description}}
			</div> -->
			<div class="overlay_summary">
				{{description}}
			</div>
		</div>
	</a>
</div>

