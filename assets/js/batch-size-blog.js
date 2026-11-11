/* Interactive companion to the adaptive batch-size draft. No charting dependencies. */
(function (global) {
  'use strict';

  var BLUE = '#1f6f9d';
  var ORANGE = '#b26631';
  var GREY = '#7b8794';
  var RED = '#ab4040';
  var COLORS = { static: GREY, dynamic: BLUE, large: ORANGE };
  var NS = 'http://www.w3.org/2000/svg';

  function clamp(x, low, high) { return Math.max(low, Math.min(high, x)); }
  function scaleFactor(r) { return 1 / Math.sqrt(r); }

  function lrShape(kind, x, c) {
    x = clamp(x, 0, 1);
    c = c === undefined ? 0.9 : c;
    if (kind === 'constant') return 1;
    if (kind === 'linear') return 1 - x;
    if (kind === 'wsd') return x <= c ? 1 : (1 - x) / (1 - c);
    return Math.pow(Math.cos(Math.PI * x / 2), 2);
  }

  function remainingLR(kind, x, c) {
    var delta = Math.max(0, 1 - x);
    c = c === undefined ? 0.9 : c;
    if (kind === 'constant') return delta;
    if (kind === 'linear') return delta * delta / 2;
    if (kind === 'wsd') return x <= c ? (1 + c) / 2 - x : delta * delta / (2 * (1 - c));
    // The direct cosine expression loses precision close to the terminal step.
    if (delta < 0.001) {
      return Math.PI * Math.PI * Math.pow(delta, 3) / 12 * (1 - Math.PI * Math.PI * delta * delta / 20);
    }
    return (delta - Math.sin(Math.PI * delta) / Math.PI) / 2;
  }

  function batchRatio(kind, x, c) {
    c = c === undefined ? 0.9 : c;
    x = clamp(x, 0, 1);
    if (kind === 'constant') return x === 1 ? Infinity : 1 / (2 * Math.sqrt(1 - x));
    if (kind === 'linear') return 1;
    if (kind === 'wsd') {
      return x <= c ? 1 / Math.sqrt((1 + c - 2 * x) * (1 + c)) : 1 / Math.sqrt(1 - c * c);
    }
    if (x === 1) return 0;
    return lrShape('cosine', x) / Math.sqrt(2 * remainingLR('cosine', x));
  }

  function cumulativeTokens(kind, x, c) {
    x = clamp(x, 0, 1);
    c = c === undefined ? 0.9 : c;
    var total = remainingLR(kind, 0, c);
    return 1 - Math.sqrt(Math.max(0, remainingLR(kind, x, c) / total));
  }

  function allocationSamples(kind, power, c, count) {
    count = count || 2000;
    c = c === undefined ? 0.9 : c;
    var raw = [], weights = [], sumBatch = 0, staticCost = 0;
    for (var i = 0; i < count; i++) {
      var x = (i + 0.5) / count;
      var b = Math.pow(batchRatio(kind, x, c), power);
      var eta = lrShape(kind, x, c);
      var w = eta * eta / remainingLR(kind, x, c);
      raw.push(b);
      weights.push(w);
      sumBatch += b;
      staticCost += w;
    }
    var average = sumBatch / count;
    var cost = 0;
    for (var j = 0; j < count; j++) cost += weights[j] / (raw[j] / average);
    return { ratio: cost / staticCost, average: average, raw: raw, weights: weights, staticCost: staticCost };
  }

  function allocationCost(kind, power, c) { return allocationSamples(kind, power, c).ratio; }

  function cumulativeAllocation(kind, power, c, count) {
    c = c === undefined ? 0.9 : c;
    var chosen = allocationSamples(kind, power, c, count);
    var derived = allocationSamples(kind, 1, c, count);
    var n = chosen.weights.length;
    var curves = { chosen: [[0, 0]], derived: [[0, 0]], static: [[0, 0]], ratio: chosen.ratio, average: chosen.average };
    var chosenSum = 0, derivedSum = 0, staticSum = 0;
    for (var i = 0; i < n; i++) {
      var weight = chosen.weights[i];
      chosenSum += weight / (chosen.raw[i] / chosen.average);
      derivedSum += weight / (derived.raw[i] / derived.average);
      staticSum += weight;
      var x = (i + 1) / n;
      curves.chosen.push([x, chosenSum / chosen.staticCost]);
      curves.derived.push([x, derivedSum / chosen.staticCost]);
      curves.static.push([x, staticSum / chosen.staticCost]);
    }
    return curves;
  }

  // Updating a controller never emits a cursor event, so linked plots cannot loop.
  function createCursorGroup(initial, min, max, onChange) {
    var current = clamp(initial, min, max), controllers = [];
    return {
      add: function (controller) {
        if (!controller) return;
        controllers.push(controller);
        controller.setCursor(current);
      },
      setCursor: function (x) {
        current = clamp(x, min, max);
        controllers.forEach(function (controller) { controller.setCursor(current); });
        if (onChange) onChange(current);
      },
      getCursor: function () { return current; }
    };
  }

  function interpolatePoint(points, x) {
    if (x <= points[0][0]) return points[0].slice();
    if (x >= points[points.length - 1][0]) return points[points.length - 1].slice();
    var left = 0, right = points.length - 1;
    while (right - left > 1) {
      var middle = Math.floor((left + right) / 2);
      if (points[middle][0] <= x) left = middle;
      else right = middle;
    }
    var a = points[left], b = points[right];
    var fraction = b[0] === a[0] ? 0 : (x - a[0]) / (b[0] - a[0]);
    return [x, a[1] + fraction * (b[1] - a[1])];
  }

  // Expose only the mathematical helpers for independent checks and reuse.
  global.BatchSizeBlogMath = {
    batchRatio: batchRatio,
    lrShape: lrShape,
    remainingLR: remainingLR,
    cumulativeTokens: cumulativeTokens,
    allocationCost: allocationCost,
    cumulativeAllocation: cumulativeAllocation,
    scaleFactor: scaleFactor,
    interpolatePoint: interpolatePoint,
    createCursorGroup: createCursorGroup
  };

  if (typeof document === 'undefined') return;
  var root = document.getElementById('batch-blog');
  if (!root) return;

  function el(tag, attrs, text) {
    var node = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (key) { node.setAttribute(key, attrs[key]); });
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function html(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function number(value, digits) {
    return Number(value).toLocaleString('en-US', { maximumFractionDigits: digits === undefined ? 2 : digits });
  }

  function compact(value) {
    if (Math.abs(value) >= 1000) return number(value / 1000, 1) + 'k';
    if (Math.abs(value) > 0 && Math.abs(value) < 0.01) return number(value, 4);
    return number(value, 2);
  }

  function output(card, name, value) {
    var node = card.querySelector('[data-' + name + ']');
    if (node) node.textContent = value;
  }

  function readout(host) {
    var node = html('div', 'batch-chart-readout');
    node.setAttribute('aria-live', 'polite');
    node.setAttribute('aria-atomic', 'true');
    host.appendChild(node);
    return node;
  }

  function runReadout(host, series) {
    var info = readout(host);
    info.classList.add('batch-chart-readout--runs');
    var position = html('div', 'batch-readout-position');
    info.appendChild(position);
    var rows = series.map(function (line, index) {
      var row = html('div', 'batch-readout-row');
      row.setAttribute('data-run', line.id || 'run-' + index);
      var swatch = html('i', 'batch-legend-swatch');
      swatch.style.backgroundColor = line.color;
      swatch.setAttribute('aria-hidden', 'true');
      row.appendChild(swatch);
      row.appendChild(html('span', 'batch-readout-label', line.label));
      var value = html('span', 'batch-readout-value');
      row.appendChild(value);
      info.appendChild(row);
      return { row: row, value: value };
    });
    return { position: position, rows: rows };
  }

  function width(host) { return Math.max(300, Math.min(850, host.clientWidth || 620)); }

  function chartSvg(host, chartWidth, chartHeight, title, description) {
    host.textContent = '';
    var svg = el('svg', { viewBox: '0 0 ' + chartWidth + ' ' + chartHeight, role: 'img', tabindex: '0', 'aria-label': title, 'class': 'batch-svg', 'font-size': 11.5 });
    svg.appendChild(el('title', {}, title));
    svg.appendChild(el('desc', {}, description || title));
    host.appendChild(svg);
    return svg;
  }

  function lineChart(host, options) {
    if (!host) return;
    var series = options.series.filter(function (line) { return line.points && line.points.length; });
    if (!series.length) { host.textContent = 'Select at least one curve to compare.'; return; }
    var w = width(host), h = options.height || 275;
    var m = { left: w < 420 ? 48 : 55, right: 18, top: 18, bottom: 47 };
    var plotWidth = w - m.left - m.right, plotHeight = h - m.top - m.bottom;
    var points = [].concat.apply([], series.map(function (line) { return line.points; }));
    var xDomain = options.xDomain || [Math.min.apply(null, points.map(function (p) { return p[0]; })), Math.max.apply(null, points.map(function (p) { return p[0]; }))];
    var yDomain = options.yDomain;
    if (!yDomain) {
      var ys = points.map(function (p) { return p[1]; }).filter(Number.isFinite);
      var low = Math.min.apply(null, ys), high = Math.max.apply(null, ys);
      var pad = (high - low || Math.abs(high) * 0.1 || 1) * 0.08;
      yDomain = [Math.max(0, low - pad), high + pad];
    }
    var sx = function (x) { return m.left + (x - xDomain[0]) / (xDomain[1] - xDomain[0]) * plotWidth; };
    var sy = function (y) { return m.top + (yDomain[1] - y) / (yDomain[1] - yDomain[0]) * plotHeight; };
    var svg = chartSvg(host, w, h, options.title, options.description || 'Use the arrow keys to explore chart values.');
    for (var tick = 0; tick <= 4; tick++) {
      var yValue = yDomain[0] + (yDomain[1] - yDomain[0]) * tick / 4;
      var y = sy(yValue);
      svg.appendChild(el('line', { x1: m.left, x2: w - m.right, y1: y, y2: y, 'class': 'batch-grid', stroke: '#e8edf2', 'stroke-width': 1 }));
      svg.appendChild(el('text', { x: m.left - 8, y: y + 4, 'text-anchor': 'end', 'class': 'batch-tick' }, (options.yTick || compact)(yValue)));
      var xValue = xDomain[0] + (xDomain[1] - xDomain[0]) * tick / 4;
      svg.appendChild(el('text', { x: sx(xValue), y: h - m.bottom + 19, 'text-anchor': 'middle', 'class': 'batch-tick' }, (options.xTick || compact)(xValue)));
    }
    svg.appendChild(el('line', { x1: m.left, x2: w - m.right, y1: h - m.bottom, y2: h - m.bottom, 'class': 'batch-axis', stroke: '#b6c3cf', 'stroke-width': 1 }));
    svg.appendChild(el('text', { x: m.left + plotWidth / 2, y: h - 5, 'text-anchor': 'middle', 'class': 'batch-axis-label' }, options.xLabel));
    svg.appendChild(el('text', { transform: 'translate(13 ' + (m.top + plotHeight / 2) + ') rotate(-90)', 'text-anchor': 'middle', 'class': 'batch-axis-label' }, options.yLabel));
    if (options.horizontalReference && options.horizontalReference.value >= yDomain[0] && options.horizontalReference.value <= yDomain[1]) {
      var referenceY = sy(options.horizontalReference.value);
      svg.appendChild(el('line', { x1: m.left, x2: w - m.right, y1: referenceY, y2: referenceY, stroke: '#929ba5', 'stroke-width': 1.1, 'stroke-dasharray': '5 4', 'class': 'batch-reference' }));
      svg.appendChild(el('text', { x: m.left + 6, y: referenceY - 7, 'class': 'batch-reference-label', 'font-size': 9.5 }, options.horizontalReference.label));
    }
    series.forEach(function (line) {
      var valid = line.points.filter(function (p) { return Number.isFinite(p[0]) && Number.isFinite(p[1]); });
      var d = valid.map(function (p, index) { return (index ? 'L' : 'M') + sx(p[0]).toFixed(2) + ',' + sy(p[1]).toFixed(2); }).join(' ');
      var path = el('path', { d: d, fill: 'none', stroke: line.color, 'stroke-width': line.width || 2.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' });
      if (line.dash) path.setAttribute('stroke-dasharray', '6 5');
      svg.appendChild(path);
    });
    var cursor = el('line', { y1: m.top, y2: h - m.bottom, stroke: '#66717c', 'stroke-dasharray': '3 4', 'stroke-width': 1 });
    svg.appendChild(cursor);
    var dots = series.map(function (line) {
      var dot = el('circle', { r: 4, fill: line.color, stroke: '#fff', 'stroke-width': 1.5 });
      svg.appendChild(dot);
      return dot;
    });
    var inspection = runReadout(host, series);
    var runRows = inspection.rows;
    var currentX = options.cursorX === undefined ? xDomain[0] + (xDomain[1] - xDomain[0]) * 0.6 : options.cursorX;
    function move(x, notify) {
      currentX = clamp(x, xDomain[0], xDomain[1]);
      svg.setAttribute('data-cursor-x', currentX);
      cursor.setAttribute('x1', sx(currentX));
      cursor.setAttribute('x2', sx(currentX));
      series.forEach(function (line, index) {
        var epsilon = (xDomain[1] - xDomain[0]) * 1e-9;
        var beforeCoverage = currentX < line.points[0][0] - epsilon;
        var afterCoverage = currentX > line.points[line.points.length - 1][0] + epsilon;
        if (beforeCoverage) {
          dots[index].setAttribute('visibility', 'hidden');
          runRows[index].row.setAttribute('data-coverage', 'before');
          runRows[index].value.textContent = '—';
          runRows[index].value.setAttribute('title', 'No plotted data at this position.');
          return;
        }
        var point = interpolatePoint(line.points, currentX);
        dots[index].setAttribute('visibility', 'visible');
        dots[index].setAttribute('cx', sx(point[0]));
        dots[index].setAttribute('cy', sy(point[1]));
        var formatted = options.valueFormat ? options.valueFormat(point[1]) : number(point[1], 3);
        var xText = options.hoverX ? options.hoverX(point[0]) : compact(point[0]);
        runRows[index].row.setAttribute('data-coverage', afterCoverage ? 'end' : 'within');
        runRows[index].value.textContent = (options.approximate ? '~' : '') + formatted + (afterCoverage ? ' (end)' : '');
        runRows[index].value.setAttribute('title', afterCoverage ? 'Run endpoint at ' + xText : 'Value at the inspected position.');
      });
      inspection.position.textContent = options.hoverX ? options.hoverX(currentX) : compact(currentX);
      if (notify && options.onCursorChange) options.onCursorChange(currentX);
    }
    svg.addEventListener('pointermove', function (event) {
      var rect = svg.getBoundingClientRect();
      var chartX = (event.clientX - rect.left) * w / rect.width;
      move(xDomain[0] + (chartX - m.left) / plotWidth * (xDomain[1] - xDomain[0]), true);
    });
    svg.addEventListener('keydown', function (event) {
      if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].indexOf(event.key) < 0) return;
      event.preventDefault();
      if (event.key === 'Home') move(xDomain[0], true);
      else if (event.key === 'End') move(xDomain[1], true);
      else move(currentX + (event.key === 'ArrowRight' ? 1 : -1) * (xDomain[1] - xDomain[0]) / 100, true);
    });
    move(currentX);
    return {
      setCursor: function (x) { move(x, false); },
      getCursor: function () { return currentX; }
    };
  }

  function sampled(fn, end, count) {
    count = count || 160;
    var points = [];
    for (var i = 0; i <= count; i++) {
      var x = (end === undefined ? 1 : end) * i / count;
      points.push([x, fn(x)]);
    }
    return points;
  }

  function bind(card, render, skip) {
    if (!card) return;
    card.querySelectorAll('input, select').forEach(function (control) {
      if (skip && control.matches(skip)) return;
      control.addEventListener(control.tagName === 'SELECT' ? 'change' : 'input', render);
    });
  }

  function errorMessage(host, message) {
    if (!host) return;
    host.textContent = '';
    host.appendChild(html('p', 'batch-error', message));
  }

  fetch(root.getAttribute('data-source'))
    .then(function (response) { if (!response.ok) throw new Error('HTTP ' + response.status); return response.json(); })
    .then(function (data) {
      var resultsCard = document.getElementById('batch-results');
      var scheduleCard = document.getElementById('batch-schedule');
      var allocationCard = document.getElementById('batch-allocation');
      var scaleCard = document.getElementById('batch-scale');
      var benchmarksCard = document.getElementById('batch-benchmarks');
      var ratioCard = document.getElementById('batch-ratio');
      var scheduleCursor, allocationCursor;
      var allocationProgress = 0.5;

      function renderResults() {
        if (!resultsCard) return;
        var model = data.training[resultsCard.querySelector('[data-result-model]').value];
        var chosen = Array.from(resultsCard.querySelectorAll('[data-series-toggle]:checked')).map(function (input) { return input.value; });
        var yDomain = [Math.min(model.axes.steps.yDomain[0], model.axes.tokens.yDomain[0]), Math.max(model.axes.steps.yDomain[1], model.axes.tokens.yDomain[1])];
        var staticLine = model.axes.steps.series.find(function (line) { return line.id === 'static'; });
        var referenceLoss = staticLine.points[staticLine.points.length - 1][1];
        var reference = chosen.indexOf('static') >= 0 && chosen.indexOf('large') >= 0 ? { value: referenceLoss, label: 'Reference loss (~' + number(referenceLoss, 2) + ')' } : null;
        ['steps', 'tokens'].forEach(function (axis) {
          var dataset = model.axes[axis];
          var allPoints = [].concat.apply([], dataset.series.map(function (line) { return line.points; }));
          var xValues = allPoints.map(function (point) { return point[0]; });
          var xDomain = axis === 'steps' ? [0, Math.ceil(Math.max.apply(null, xValues) / 1000) * 1000] : [Math.min.apply(null, xValues), Math.max.apply(null, xValues)];
          var series = dataset.series.filter(function (line) { return chosen.indexOf(line.id) >= 0; }).map(function (line) {
            var labels = { static: 'Static', dynamic: 'Dynamic', large: 'Dynamic (large batch)' };
            return { id: line.id, label: labels[line.id] || line.label, points: line.points, color: COLORS[line.id] || BLUE, width: line.id === 'large' ? 3.2 : 2.2 };
          });
          lineChart(resultsCard.querySelector('[data-result-' + axis + '-chart]'), {
            title: model.label + ' test loss versus ' + (axis === 'steps' ? 'optimizer steps' : 'training tokens') + ' (rough visual reconstruction)',
            series: series, xDomain: xDomain, yDomain: yDomain, xLabel: dataset.xLabel, yLabel: 'Test loss (lower is better)', approximate: true,
            horizontalReference: reference,
            valueFormat: function (y) { return number(y, 2); },
            hoverX: function (x) { return axis === 'tokens' ? number(x, 2) + 'B tokens' : number(x, 0) + ' steps'; }
          });
        });
        output(resultsCard, 'result-steps-note', 'Figure 1 reports ' + model.iterationReduction + '% fewer optimizer updates for the larger-average-batch run at the reference loss.');
        output(resultsCard, 'result-tokens-note', 'The larger-average-batch run reaches the reference loss with a comparable number of training tokens.');
      }

      function renderSchedule() {
        if (!scheduleCard) return;
        var kind = scheduleCard.querySelector('[data-schedule]').value;
        var cooldown = Number(scheduleCard.querySelector('[data-cooldown]').value);
        var c = 1 - cooldown / 100;
        var average = Number(scheduleCard.querySelector('[data-average]').value);
        var peakMultiplier = Number(scheduleCard.querySelector('[data-peak]').value);
        var peak = 0.01 * peakMultiplier;
        var progressInput = scheduleCard.querySelector('[data-progress]');
        var end = kind === 'constant' ? 0.995 : 1;
        var progress = clamp(Number(progressInput.value) / 100, 0, end);
        progressInput.max = String(end * 100);
        progressInput.step = '0.1';
        scheduleCard.querySelector('[data-cooldown]').disabled = kind !== 'wsd';
        output(scheduleCard, 'cooldown-value', cooldown + '%');
        output(scheduleCard, 'peak-value', number(peakMultiplier, 2) + '×');
        var pct = function (x) { return number(x * 100, 0) + '%'; };
        var inspectPct = function (x) { return number(x * 100, 1) + '%'; };
        var trainingPosition = function (x) { return inspectPct(x) + ' training'; };
        scheduleCursor = createCursorGroup(progress, 0, end, function (x) {
          progressInput.value = String(x * 100);
          output(scheduleCard, 'progress-value', inspectPct(x));
          output(scheduleCard, 'batch-value', number(average * batchRatio(kind, x, c), 0));
          output(scheduleCard, 'lr-value', number(peak * lrShape(kind, x, c), 6));
          output(scheduleCard, 'budget-value', number(cumulativeTokens(kind, x, c) * 100, 1) + '%');
        });
        function syncSchedule(x) { scheduleCursor.setCursor(x); }
        scheduleCursor.add(lineChart(scheduleCard.querySelector('[data-lr-chart]'), {
          title: 'Selected learning-rate schedule', xLabel: 'Training progress (steps)', yLabel: 'Learning rate',
          xDomain: [0, 1], yDomain: [0, Math.max(peak * 1.05, 0.003)], xTick: pct, hoverX: trainingPosition, cursorX: progress, onCursorChange: syncSchedule,
          valueFormat: function (y) { return number(y, 6); },
          series: [{ id: 'lr', label: 'Learning rate', color: ORANGE, points: sampled(function (x) { return peak * lrShape(kind, x, c); }) }]
        }));
        scheduleCursor.add(lineChart(scheduleCard.querySelector('[data-batch-shape-chart]'), {
          title: 'The derived batch-size allocation', xLabel: 'Training progress (steps)', yLabel: 'Batch / average batch',
          xDomain: [0, 1], xTick: pct, hoverX: trainingPosition, cursorX: progress, onCursorChange: syncSchedule,
          valueFormat: function (y) { return number(y, 3) + '×'; },
          series: [
            { id: 'derived', label: 'Derived', color: BLUE, points: sampled(function (x) { return batchRatio(kind, x, c); }, end) },
            { id: 'static', label: 'Static', color: GREY, dash: true, points: [[0, 1], [1, 1]] }
          ]
        }));
        scheduleCursor.add(lineChart(scheduleCard.querySelector('[data-token-chart]'), {
          title: 'Cumulative token budget', height: 220, xLabel: 'Training progress (steps)', yLabel: 'Token budget consumed',
          xDomain: [0, 1], yDomain: [0, 1.03], xTick: pct, yTick: pct, hoverX: trainingPosition, cursorX: progress, onCursorChange: syncSchedule,
          valueFormat: inspectPct,
          series: [
            { id: 'derived', label: 'Derived', color: BLUE, points: sampled(function (x) { return cumulativeTokens(kind, x, c); }) },
            { id: 'static', label: 'Static', color: GREY, dash: true, points: [[0, 0], [1, 1]] }
          ]
        }));
        scheduleCursor.setCursor(progress);
      }

      function renderAllocation() {
        if (!allocationCard) return;
        var kind = allocationCard.querySelector('[data-allocation-schedule]').value;
        var power = Number(allocationCard.querySelector('[data-allocation-power]').value);
        var cumulative = cumulativeAllocation(kind, power, 0.9);
        output(allocationCard, 'allocation-power-value', number(power, 2));
        output(allocationCard, 'allocation-score', number(cumulative.ratio, 3) + '× full-run static noise term');
        var pct = function (x) { return number(x * 100, 0) + '%'; };
        var inspectPct = function (x) { return number(x * 100, 1) + '%'; };
        var trainingPosition = function (x) { return inspectPct(x) + ' training'; };
        allocationCursor = createCursorGroup(allocationProgress, 0, 1, function (x) {
          allocationProgress = x;
          output(allocationCard, 'allocation-progress-value', inspectPct(x));
        });
        function syncAllocation(x) { allocationCursor.setCursor(x); }
        allocationCursor.add(lineChart(allocationCard.querySelector('[data-allocation-batches]'), {
          title: 'Redistribute the same token budget', xLabel: 'Training progress (steps)', yLabel: 'Batch / average batch',
          xDomain: [0, 1], xTick: pct, hoverX: trainingPosition, cursorX: allocationProgress, onCursorChange: syncAllocation,
          valueFormat: function (y) { return number(y, 3) + '×'; },
          series: [
            { id: 'selected', label: 'Selected', color: ORANGE, points: sampled(function (x) { return Math.pow(batchRatio(kind, x, 0.9), power) / cumulative.average; }) },
            { id: 'optimal', label: 'Optimal', color: BLUE, dash: true, points: sampled(function (x) { return batchRatio(kind, x, 0.9); }) }
          ]
        }));
        allocationCursor.add(lineChart(allocationCard.querySelector('[data-allocation-cost]'), {
          title: 'Cumulative weighted-noise contribution, not measured LLM loss',
          xLabel: 'Training progress (steps)', yLabel: 'Noise / full static total', xDomain: [0, 1], cursorX: allocationProgress,
          xTick: pct, hoverX: trainingPosition, onCursorChange: syncAllocation,
          valueFormat: function (y) { return number(y, 3) + '×'; },
          series: [
            { id: 'selected', label: 'Selected', color: ORANGE, points: cumulative.chosen },
            { id: 'optimal', label: 'Optimal', color: BLUE, points: cumulative.derived },
            { id: 'static', label: 'Static', color: GREY, dash: true, points: cumulative.static }
          ]
        }));
        allocationCursor.setCursor(allocationProgress);
      }

      function renderScale() {
        if (!scaleCard) return;
        var r = Number(scaleCard.querySelector('[data-horizon]').value);
        output(scaleCard, 'horizon-value', number(r, 0) + '×');
        output(scaleCard, 'scale-rate', number(0.01 * scaleFactor(r), 6));
        output(scaleCard, 'scale-decay', number(0.1 * scaleFactor(r), 6));
        lineChart(scaleCard.querySelector('[data-scaling-chart]'), {
          title: 'Scale the learning rate and weight decay with the horizon',
          xLabel: 'Training-horizon multiplier', yLabel: 'Multiplier for LR and decay', xDomain: [1, 16], yDomain: [0, 1.05], cursorX: r,
          hoverX: function (x) { return number(x, 1) + '× horizon'; },
          valueFormat: function (y) { return number(y, 3) + '×'; },
          series: [{ id: 'scale', label: 'LR & decay', color: BLUE, points: Array.from({ length: 151 }, function (_, i) { var x = 1 + i / 10; return [x, scaleFactor(x)]; }) }]
        });
      }

      function renderBenchmarks() {
        if (!benchmarksCard) return;
        var model = data.benchmarks.models.find(function (item) { return item.id === benchmarksCard.querySelector('[data-benchmark-model]').value; });
        var host = benchmarksCard.querySelector('[data-benchmark-chart]');
        var w = width(host), h = 375, left = w < 420 ? 108 : 120, right = 52, top = 20, row = 36;
        var deltas = model.dynamic.map(function (value, i) { return value - model.static[i]; });
        var min = Math.min(-0.5, Math.min.apply(null, deltas) - 0.15), max = Math.max.apply(null, deltas) + 0.3;
        var sx = function (x) { return left + (x - min) / (max - min) * (w - left - right); };
        var svg = chartSvg(host, w, h, model.label + ' benchmark score changes', 'Dynamic minus static accuracy in percentage points. Use up and down arrow keys to inspect each task.');
        [min, 0, max].forEach(function (tick) {
          svg.appendChild(el('line', { x1: sx(tick), x2: sx(tick), y1: 10, y2: top + row * 8, 'class': tick === 0 ? 'batch-axis' : 'batch-grid', stroke: tick === 0 ? '#b6c3cf' : '#e8edf2', 'stroke-width': 1 }));
          svg.appendChild(el('text', { x: sx(tick), y: top + row * 8 + 19, 'text-anchor': 'middle', 'class': 'batch-tick' }, number(tick, 1)));
        });
        var markers = [];
        data.benchmarks.tasks.forEach(function (task, i) {
          var y = top + i * row + 14, delta = deltas[i], color = delta >= 0 ? BLUE : RED;
          svg.appendChild(el('text', { x: left - 10, y: y + 4, 'text-anchor': 'end', 'class': 'batch-tick' }, task));
          svg.appendChild(el('line', { x1: sx(0), x2: sx(delta), y1: y, y2: y, stroke: color, 'stroke-width': 4 }));
          var dot = el('circle', { cx: sx(delta), cy: y, r: 4, fill: color });
          svg.appendChild(dot);
          markers.push(dot);
          svg.appendChild(el('text', { x: w - right + 10, y: y + 4, 'class': 'batch-tick', fill: color }, (delta >= 0 ? '+' : '') + delta.toFixed(2)));
        });
        svg.appendChild(el('text', { x: left + (w - left - right) / 2, y: h - 10, 'text-anchor': 'middle', 'class': 'batch-axis-label' }, 'Dynamic − static (percentage points)'));
        var inspection = runReadout(host, [
          { id: 'static', label: 'Static', color: GREY },
          { id: 'dynamic', label: 'Dynamic', color: BLUE }
        ]);
        var selected = 0;
        function select(index) {
          selected = clamp(index, 0, 7);
          markers.forEach(function (dot, i) { dot.setAttribute('r', i === selected ? 6 : 4); });
          inspection.position.textContent = data.benchmarks.tasks[selected] + ' · ' + (deltas[selected] >= 0 ? '+' : '') + deltas[selected].toFixed(2) + ' pp';
          inspection.rows[0].value.textContent = model.static[selected].toFixed(2) + '%';
          inspection.rows[1].value.textContent = model.dynamic[selected].toFixed(2) + '%';
        }
        svg.addEventListener('pointermove', function (event) {
          var rect = svg.getBoundingClientRect();
          select(Math.round(((event.clientY - rect.top) * h / rect.height - top - 14) / row));
        });
        svg.addEventListener('keydown', function (event) {
          if (['ArrowUp', 'ArrowDown', 'Home', 'End'].indexOf(event.key) < 0) return;
          event.preventDefault();
          select(event.key === 'Home' ? 0 : event.key === 'End' ? 7 : selected + (event.key === 'ArrowDown' ? 1 : -1));
        });
        select(0);
        output(benchmarksCard, 'benchmark-note', model.label + ': dynamic batching improves ' + deltas.filter(function (d) { return d > 0; }).length + ' of 8 task scores in Table 1. Values are transcribed from the table, not estimated from plots.');
      }

      function renderRatio() {
        if (!ratioCard) return;
        var lr = Number(ratioCard.querySelector('[data-ratio-lr]').value);
        var schedule = ratioCard.querySelector('[data-ratio-schedule]').value;
        var result = data.ratio.rows.find(function (item) { return Math.abs(item.lr - lr) < 1e-10 && item.schedule === schedule; });
        if (!result) throw new Error('The selected matched-ratio result is missing.');
        var host = ratioCard.querySelector('[data-ratio-chart]');
        var w = width(host), h = 190, left = w < 420 ? 105 : 135, right = 53;
        var max = Math.max(result.lrLoss, result.batchLoss) * 1.08;
        var sx = function (x) { return left + x / max * (w - left - right); };
        var svg = chartSvg(host, w, h, 'Same learning-rate-to-batch ratio, different final loss', 'Both configurations share a matched learning-rate-to-batch-size ratio and token budget. Lower validation loss is better.');
        var rows = [{ label: 'LR schedule', value: result.lrLoss, color: BLUE }, { label: 'Batch schedule', value: result.batchLoss, color: ORANGE }];
        rows.forEach(function (item, i) {
          var y = 25 + i * 58;
          svg.appendChild(el('text', { x: left - 10, y: y + 19, 'text-anchor': 'end', 'class': 'batch-tick' }, item.label));
          var bar = el('rect', { x: left, y: y, width: sx(item.value) - left, height: 28, rx: 3, fill: item.color });
          bar.appendChild(el('title', {}, item.label + ': ' + item.value.toFixed(3)));
          svg.appendChild(bar);
          svg.appendChild(el('text', { x: sx(item.value) + 8, y: y + 19, 'class': 'batch-tick' }, item.value.toFixed(3)));
        });
        [0, 1, 2, 3, 4].forEach(function (i) { var value = max * i / 4; svg.appendChild(el('text', { x: sx(value), y: h - 37, 'text-anchor': 'middle', 'class': 'batch-tick' }, number(value, 1))); });
        svg.appendChild(el('text', { x: (left + w - right) / 2, y: h - 10, 'text-anchor': 'middle', 'class': 'batch-axis-label' }, 'Final validation loss (lower is better)'));
        var gap = result.reportedDelta === undefined ? result.batchLoss - result.lrLoss : result.reportedDelta;
        output(ratioCard, 'ratio-note', 'Table 4 reports ' + gap.toFixed(3) + ' higher loss for the batch-scheduled run, despite the matched ratio and token budget. Endpoints are rounded as printed; the reported gap can differ slightly from their subtraction.');
      }

      var renders = [renderResults, renderSchedule, renderAllocation, renderScale, renderBenchmarks, renderRatio];
      function drawAll() { renders.forEach(function (render) { render(); }); }
      bind(resultsCard, renderResults);
      bind(scheduleCard, renderSchedule, '[data-progress]');
      if (scheduleCard) scheduleCard.querySelector('[data-progress]').addEventListener('input', function (event) {
        if (scheduleCursor) scheduleCursor.setCursor(Number(event.target.value) / 100);
      });
      bind(allocationCard, renderAllocation);
      bind(scaleCard, renderScale);
      bind(benchmarksCard, renderBenchmarks);
      bind(ratioCard, renderRatio);
      root.querySelectorAll('details').forEach(function (node) {
        node.addEventListener('toggle', function () {
          drawAll();
          if (node.open && global.MathJax && global.MathJax.Hub && typeof global.MathJax.Hub.Queue === 'function') {
            global.MathJax.Hub.Queue(['Rerender', global.MathJax.Hub, node]);
          }
        });
      });
      var resizeTimer;
      global.addEventListener('resize', function () { clearTimeout(resizeTimer); resizeTimer = setTimeout(drawAll, 120); });
      drawAll();
      root.setAttribute('data-ready', 'true');
    })
    .catch(function (error) {
      root.setAttribute('data-error', 'true');
      root.querySelectorAll('.batch-chart').forEach(function (host) { errorMessage(host, 'The interactive figures could not load. Please refresh the page or read the linked paper.'); });
      if (global.console) global.console.error('Batch-size blog:', error);
    });
}(typeof window !== 'undefined' ? window : globalThis));
