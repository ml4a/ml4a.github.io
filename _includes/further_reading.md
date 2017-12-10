{% assign title = include.title %}
{% assign author = include.author %}
{% assign link = include.link %}
{% assign description = include.description %}

<style>
	.further_reading{
		margin-left: auto;
		margin-right: auto;
		width:720px;
		margin-bottom:20px;
		margin-top:20px;
		border: 1px solid #ccc;		
	}
	.further_reading .fr_header{
		background-color:#252;
		font-size:1.5em;
		padding:4px;
		margin:2px;
		color: #fff;
	}
	.further_reading .fr_content{
		font-size:1.3em;
		padding:6px;
		line-height:150%;
	}
</style>

<div class="further_reading">
	<div class="fr_header">
		Further reading
	</div>
	<div class="fr_content">
		<a href="{{ link }}">{{ title }}</a> by {{ author }}
	</div>
</div>
