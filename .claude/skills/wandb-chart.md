---
name: wandb-chart
description: Fetch wandb run data and generate an interactive Plotly chart for a blog post
user_invocable: true
---

# wandb-chart skill

Generate an interactive loss curve chart from Weights & Biases runs and embed it in a blog post.

## Workflow

When the user invokes `/wandb-chart`, ask them for the following (or extract from their message):

1. **entity** — wandb entity (default: `jasonljx96`)
2. **project** — wandb project name (e.g., `modded-nanogpt`)
3. **run IDs** — list of wandb run IDs to plot
4. **run names** — display names for each run in the legend
5. **metric** — metric key to plot (default: `val_loss`)
6. **output name** — JSON filename for the data (e.g., `my-chart.json`)
7. **blog post** — which blog post file to insert the chart into (optional — user may want to copy-paste manually)
8. **figure number and caption** — e.g., "Figure 5: Validation loss comparison"
9. **styling** — per-run color, line width, dash style (optional, use sensible defaults)

## Steps

### Step 1: Fetch data from wandb

Run the fetch script from the project root:

```bash
python _blog/blog_tests/fetch_wandb.py \
  --entity <entity> \
  --project <project> \
  --runs <run_id_1> <run_id_2> ... \
  --metric <metric> \
  --output assets/data/<output_name>.json
```

The script has a default WANDB_API_KEY built in. The user can override it with the `WANDB_API_KEY` env var.

### Step 2: Generate the HTML snippet

Insert the following into the blog post at the desired location:

```html
<figure id="fig-<id>" style="text-align: center;">
  <div id="<chart-id>"></div>
  <figcaption style="margin-top: 1em;">Figure N: Caption text.</figcaption>
</figure>
<script src="/assets/js/wandb-chart.js"></script>
<script>
renderWandbChart('<chart-id>', '/assets/data/<output_name>.json', {
  xLabel: 'Step',
  yLabel: '<metric display name>',
  height: 400,
  runs: [
    { id: '<run_id_1>', name: '<Display Name 1>', color: '#1f77b4', width: 2, dash: 'solid' },
    { id: '<run_id_2>', name: '<Display Name 2>', color: '#ff7f0e', width: 2, dash: 'dash' },
  ]
});
</script>
```

### Per-run styling options

| Option  | Values                                  | Default    |
|---------|-----------------------------------------|------------|
| `color` | Any CSS color (e.g., `#1f77b4`, `red`)  | auto       |
| `width` | Line width in px                        | `2`        |
| `dash`  | `solid`, `dash`, `dot`, `dashdot`       | `solid`    |
| `name`  | Legend display name                     | run name   |

### Default color palette

Use these colors in order for multiple runs:
- `#1f77b4` (blue)
- `#ff7f0e` (orange)
- `#2ca02c` (green)
- `#d62728` (red)
- `#9467bd` (purple)
- `#8c564b` (brown)

### Chart config options

| Option   | Description              | Default          |
|----------|--------------------------|------------------|
| `xLabel` | X-axis label             | `Step`           |
| `yLabel` | Y-axis label             | metric name      |
| `height` | Chart height in px       | `450`            |

## Key files

- `_blog/blog_tests/fetch_wandb.py` — Data fetcher script
- `assets/js/wandb-chart.js` — Reusable Plotly chart renderer (loaded once per page)
- `assets/data/` — Directory for chart JSON data files

## Notes

- The `wandb-chart.js` script only needs to be included once per page (multiple charts can share it)
- Plotly.js (~3MB) is loaded from CDN on first use
- The JSON data files must be committed to git (they are served as static assets)
- Charts support hover-to-see-values, zoom, and pan out of the box
