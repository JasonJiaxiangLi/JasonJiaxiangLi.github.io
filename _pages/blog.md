---
layout: archive
title: "Blog"
permalink: /blog/
author_profile: false
---

{% include base_path %}

{% assign sorted_posts = site.blog | sort: 'date' | reverse %}

{% for post in sorted_posts %}
  {% if post.hidden %}{% continue %}{% endif %}
  <div class="blog-listing__item">
    <h2 class="archive__item-title">
      {% if post.external_url %}
        <a href="{{ post.external_url }}">{{ post.title }}</a>
      {% else %}
        <a href="{{ base_path }}{{ post.url }}">{{ post.title }}</a>
      {% endif %}
    </h2>
    {% if post.external_url %}
    <p class="page__meta">
      By <a href="{{ post.external_author_url }}">{{ post.external_author }}</a> &middot; External post &middot; Shared {{ post.date | date: "%B %-d, %Y" }}
    </p>
    {% else %}
    <p class="page__meta">
      <i class="fa fa-fw fa-calendar" aria-hidden="true"></i>
      {{ post.date | date: "%B %d, %Y" }}
    </p>
    {% endif %}
    {% if post.excerpt %}
      <p class="archive__item-excerpt">{{ post.excerpt | markdownify | strip_html | truncatewords: 50 }}</p>
    {% endif %}
  </div>
{% endfor %}
