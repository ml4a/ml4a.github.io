{% assign path1 = include.path1 %}
{% assign caption1 = include.caption1 %}
{% assign path2 = include.path2 %}
{% assign caption2 = include.caption2 %}
{% assign path3 = include.path3 %}
{% assign caption3 = include.caption3 %}

<div class="figure_multi">
	{% if path1 %}
	<div class="figure_inner">
		<figure>
		    <img src="{{path1}}" alt="" />
			<figcaption>{{caption1}}</figcaption>
		</figure>
	</div>
	{% endif %}
	{% if path2 %}
	<div class="figure_inner">
		<figure>
		    <img src="{{path2}}" alt="" />
			<figcaption>{{caption2}}</figcaption>
		</figure>
	</div>
	{% endif %}
	{% if path3 %}
	<div class="figure_inner">
		<figure>
		    <img src="{{path3}}" alt="" />
			<figcaption>{{caption3}}</figcaption>
		</figure>
	</div>
	{% endif %}
</div>