---
layout: default
---

<h1>{{ page.title }}</h1>
<div id="post" class="post">

	{% if page.translator %}
	<p>
	<span class="translator_credit">
		translated by <a href="{{page.translator_link}}">{{ page.translator }}</a>
	</span>
	</p>
	{% endif %}

	{{ content }}

</div>
