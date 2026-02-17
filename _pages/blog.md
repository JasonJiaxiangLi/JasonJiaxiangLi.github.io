---
layout: archive
title: "Blog"
permalink: /blog/
author_profile: true
---

{% include base_path %}

{% assign sorted_posts = site.blog | sort: 'date' | reverse %}

{% for post in sorted_posts %}
  {% if post.hidden %}{% continue %}{% endif %}
  <div class="blog-listing__item">
    <h2 class="archive__item-title">
      <a href="{{ base_path }}{{ post.url }}">{{ post.title }}</a>
    </h2>
    <p class="page__meta">
      <i class="fa fa-fw fa-calendar" aria-hidden="true"></i>
      {{ post.date | date: "%B %d, %Y" }}
    </p>
    {% if post.excerpt %}
      <p class="archive__item-excerpt">{{ post.excerpt | markdownify | strip_html | truncatewords: 50 }}</p>
    {% endif %}
  </div>
{% endfor %}
