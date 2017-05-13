{% assign path1 = include.path1 %}
{% assign caption1 = include.caption1 %}
{% assign path2 = include.path2 %}
{% assign caption2 = include.caption2 %}
{% assign path3 = include.path3 %}
{% assign caption3 = include.caption3 %}

<div class="figure_outer">
	{% if path1 %}
	<div class="figure_insert">
		<img src="{{path1}}" />
		<div class="figure_caption">
			{{caption1}}
		</div>
	</div>
	{% endif %}
	{% if path2 %}
	<div class="figure_insert">
		<img src="{{path2}}" />
		<div class="figure_caption">
			{{caption2}}
		</div>
	</div>
	{% endif %}
	{% if path3 %}
	<div class="figure_insert">
		<img src="{{path3}}" />
		<div class="figure_caption">
			{{caption3}}
		</div>
	</div>
	{% endif %}
</div>

