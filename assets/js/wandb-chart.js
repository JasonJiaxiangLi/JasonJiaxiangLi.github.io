/**
 * Interactive wandb chart for blog posts using Plotly.js.
 *
 * Usage in a blog post:
 *
 *   <div id="my-chart"></div>
 *   <script src="/assets/js/wandb-chart.js"></script>
 *   <script>
 *   renderWandbChart('my-chart', '/assets/data/my-data.json', {
 *     title: 'Validation Loss',
 *     xLabel: 'Step',
 *     yLabel: 'Val Loss',
 *     height: 450,
 *     runs: [
 *       { id: 'run_id_1', name: 'Baseline', color: '#1f77b4', width: 2, dash: 'solid' },
 *       { id: 'run_id_2', name: 'Ours',     color: '#ff7f0e', width: 2, dash: 'dash' },
 *     ]
 *   });
 *   </script>
 *
 * Config options:
 *   title    — chart title (default: '')
 *   xLabel   — x-axis label (default: 'Step')
 *   yLabel   — y-axis label (default: metric name from JSON)
 *   height   — chart height in px (default: 450)
 *   runs[]   — array of run configs:
 *     id     — run ID (must match a key in the JSON)
 *     name   — display name in legend
 *     color  — line color (any CSS color)
 *     width  — line width in px (default: 2)
 *     dash   — 'solid', 'dash', 'dot', 'dashdot' (default: 'solid')
 */

(function () {
  // Load Plotly from CDN if not already present
  function ensurePlotly(callback) {
    if (window.Plotly) {
      callback();
      return;
    }
    var script = document.createElement('script');
    script.src = 'https://cdn.plot.ly/plotly-2.35.2.min.js';
    script.onload = callback;
    script.onerror = function () {
      console.error('Failed to load Plotly.js');
    };
    document.head.appendChild(script);
  }

  window.renderWandbChart = function (elementId, dataUrl, config) {
    config = config || {};
    var container = document.getElementById(elementId);
    if (!container) {
      console.error('wandb-chart: element #' + elementId + ' not found');
      return;
    }

    // Show loading state
    container.style.minHeight = (config.height || 450) + 'px';
    container.innerHTML = '<p style="text-align:center;color:#888;padding-top:2em;">Loading chart...</p>';

    ensurePlotly(function () {
      fetch(dataUrl)
        .then(function (r) { return r.json(); })
        .then(function (data) { render(container, data, config); })
        .catch(function (err) {
          container.innerHTML = '<p style="color:red;">Error loading chart data: ' + err + '</p>';
        });
    });
  };

  function render(container, data, config) {
    var runConfigs = config.runs || [];
    var traces = [];

    // If no run configs specified, plot all runs with defaults
    if (runConfigs.length === 0) {
      var defaultColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'];
      var i = 0;
      for (var rid in data.runs) {
        runConfigs.push({ id: rid, name: data.runs[rid].name, color: defaultColors[i % defaultColors.length] });
        i++;
      }
    }

    for (var j = 0; j < runConfigs.length; j++) {
      var rc = runConfigs[j];
      var runData = data.runs[rc.id];
      if (!runData) {
        console.warn('wandb-chart: run ID "' + rc.id + '" not found in data');
        continue;
      }

      traces.push({
        x: runData.steps,
        y: runData.values,
        type: 'scatter',
        mode: 'lines',
        name: rc.name || runData.name,
        line: {
          color: rc.color || undefined,
          width: rc.width || 2,
          dash: rc.dash || 'solid'
        },
        hovertemplate: '%{y:.4f}<extra>' + (rc.name || runData.name) + '</extra>'
      });
    }

    var layout = {
      title: '',
      xaxis: {
        title: config.xLabel || 'Step',
        gridcolor: '#e8e8e8',
        zerolinecolor: '#e8e8e8'
      },
      yaxis: {
        title: config.yLabel || data.metric || 'Value',
        gridcolor: '#e8e8e8',
        zerolinecolor: '#e8e8e8'
      },
      height: config.height || 450,
      margin: { l: 60, r: 30, t: 10, b: 50 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', size: 13 },
      legend: {
        orientation: 'v',
        yanchor: 'top',
        y: 0.95,
        xanchor: 'right',
        x: 0.95,
        bgcolor: 'rgba(255,255,255,0.85)',
        bordercolor: '#ccc',
        borderwidth: 1,
        font: { size: 13 }
      },
      hovermode: 'x unified'
    };

    var plotConfig = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d']
    };

    container.innerHTML = '';
    Plotly.newPlot(container, traces, layout, plotConfig);
  }
})();
