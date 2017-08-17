{% assign title = include.title %}
{% assign author = include.author %}
{% assign link = include.link %}

<p class="further_reading" style="text-align:center;color:#88f;">
	<b>further reading:</b> <a href="{{ link }}">{{ title }}</a> ... {{ author }}
</p>
