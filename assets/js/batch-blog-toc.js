/* Keep this draft's contents beside the article, directly below the masthead. */
(function (global) {
  'use strict';
  if (typeof document === 'undefined') return;
  var root = document.getElementById('batch-blog');
  var main = document.getElementById('main');
  var body = document.body;
  var toc = root && root.querySelector('#markdown-toc');
  var article = main && main.querySelector('article');
  if (!root || !main || !body || !toc || !article || root.hasAttribute('data-batch-toc-ready')) return;

  var entries = [];
  toc.querySelectorAll('a[href]').forEach(function (link) {
    var href = link.getAttribute('href');
    if (!href || href.charAt(0) !== '#') return;
    var id;
    try { id = decodeURIComponent(href.slice(1)); } catch (error) { return; }
    var heading = document.getElementById(id);
    if (heading && root.contains(heading)) entries.push({ link: link, heading: heading });
  });
  if (!entries.length) return;

  var rail = document.createElement('nav');
  rail.className = 'batch-toc-rail';
  rail.setAttribute('aria-label', 'Table of contents');
  var panel = document.createElement('details');
  panel.className = 'batch-toc-panel';
  var summary = document.createElement('summary');
  summary.textContent = 'Table of contents';
  panel.appendChild(summary);
  panel.appendChild(toc);
  rail.appendChild(panel);
  main.insertBefore(rail, article);
  main.classList.add('batch-has-toc');
  body.classList.add('batch-toc-layout');
  root.setAttribute('data-batch-toc-ready', 'true');

  var masthead = document.querySelector('.masthead');
  var desktopQuery = global.matchMedia ? global.matchMedia('(min-width: 1200px)') : null;
  var previousDesktop;
  var anchorOffset = 16;
  var pending = false, needsMeasure = false;
  var previousHeader = '', previousOffset = '';
  var requestFrame = global.requestAnimationFrame ? global.requestAnimationFrame.bind(global) : function (callback) { return global.setTimeout(callback, 16); };

  function isDesktop() { return desktopQuery ? desktopQuery.matches : global.innerWidth >= 1200; }

  function responsiveState() {
    var desktop = isDesktop();
    if (desktop !== previousDesktop) {
      panel.open = desktop;
      previousDesktop = desktop;
    }
  }

  function measure() {
    var headerHeight = masthead ? Math.max(0, Math.ceil(masthead.getBoundingClientRect().height)) : 0;
    var railHeight = isDesktop() ? 0 : Math.max(0, Math.ceil(rail.getBoundingClientRect().height));
    anchorOffset = headerHeight + railHeight + 16;
    var headerValue = headerHeight + 'px', offsetValue = anchorOffset + 'px';
    if (headerValue !== previousHeader) {
      body.style.setProperty('--batch-header-height', headerValue);
      previousHeader = headerValue;
    }
    if (offsetValue !== previousOffset) {
      body.style.setProperty('--batch-anchor-offset', offsetValue);
      previousOffset = offsetValue;
    }
  }

  function updateActive() {
    var active = -1;
    entries.forEach(function (entry, index) {
      if (entry.heading.getClientRects && !entry.heading.getClientRects().length) return;
      if (entry.heading.getBoundingClientRect().top <= anchorOffset + 1) active = index;
    });
    entries.forEach(function (entry, index) {
      if (index === active) {
        entry.link.classList.add('active-toc');
        entry.link.setAttribute('aria-current', 'location');
      } else {
        entry.link.classList.remove('active-toc');
        entry.link.removeAttribute('aria-current');
      }
    });
  }

  function refresh(withMeasure) {
    needsMeasure = needsMeasure || withMeasure;
    if (pending) return;
    pending = true;
    requestFrame(function () {
      pending = false;
      if (needsMeasure) {
        needsMeasure = false;
        responsiveState();
        measure();
      }
      updateActive();
    });
  }

  // The theme binds a -20px smooth-scroll offset to all links. Intercept only
  // this rail's ordinary anchor clicks so its measured sticky offset is used.
  rail.addEventListener('click', function (event) {
    if (event.button > 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    var link = event.target.closest && event.target.closest('a[href]');
    var entry = entries.find(function (item) { return item.link === link; });
    if (!entry) return;
    event.preventDefault();
    event.stopPropagation();
    responsiveState();
    measure();
    var top = Math.max(0, (global.scrollY || global.pageYOffset || 0) + entry.heading.getBoundingClientRect().top - anchorOffset);
    var reducedMotion = global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches;
    global.scrollTo({ top: top, behavior: reducedMotion ? 'auto' : 'smooth' });
    var hash = link.getAttribute('href');
    if (global.history && global.history.pushState && (!global.location || global.location.hash !== hash)) {
      global.history.pushState(null, '', hash);
    }
    refresh(false);
  }, true);

  global.addEventListener('scroll', function () { refresh(false); }, { passive: true });
  global.addEventListener('resize', function () { refresh(true); });
  panel.addEventListener('toggle', function () { refresh(true); });
  if (desktopQuery && desktopQuery.addEventListener) desktopQuery.addEventListener('change', function () { refresh(true); });
  else if (desktopQuery && desktopQuery.addListener) desktopQuery.addListener(function () { refresh(true); });
  if (typeof global.ResizeObserver === 'function') {
    var observer = new global.ResizeObserver(function () { refresh(true); });
    if (masthead) observer.observe(masthead);
    observer.observe(rail);
  }
  responsiveState();
  measure();
  updateActive();
}(typeof window !== 'undefined' ? window : globalThis));
