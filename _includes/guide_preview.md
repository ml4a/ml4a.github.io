<!-- {% assign page = site.works | where:"title", include.title | first %} -->

{% assign name = include.name %}
{% assign guide = site.data.guides[name] %}
{% assign title = guide.title %}
{% assign description = guide.description %}
{% assign category = guide.category %}
{% assign summary = guide.summary %}
{% assign link = guide.link %}
{% assign thumb = guide.thumb %}


<div class="project {{category}}">
	<a href="{{link}}">
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

