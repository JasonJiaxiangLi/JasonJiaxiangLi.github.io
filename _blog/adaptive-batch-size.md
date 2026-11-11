---
title: "Your batch size should have a schedule, too"
date: 2026-11-10
tags: [llm, optimization, training]
excerpt: "TL;DR: Let your learning-rate schedule determine your batch-size schedule."
hidden: false
---

<link rel="stylesheet" href="{{ '/assets/css/batch-size-blog.css' | relative_url }}">

<div id="batch-blog" class="batch-blog" data-source="{{ '/assets/data/batch-size-draft.json' | relative_url }}" markdown="1">

<p class="batch-tldr"><strong>TL;DR.</strong> Batch size should be proportional to the <strong>current learning rate divided by the square root of the remaining area under the learning-rate curve</strong>, then normalized to keep the total token budget fixed. The resulting schedule depends on the LR shape, not its peak value.</p>

<p class="batch-intro">We warm up the learning rate. We decay it. Yet we often keep the batch size fixed. Could the same training tokens do more useful work if we changed when we spent them?</p>

In our [paper](https://arxiv.org/abs/2607.27731), with **Zhiqi Bu and Shiyun Xu**, we derive a batch-size schedule from a planned learning-rate schedule, then test it on language and vision-language models. The idea is to **redistribute a fixed budget across updates**, rather than simply use more data.

<div class="batch-highlights">
  <div class="batch-stat"><strong>+15%</strong><small>Reported compute efficiency<br>Llama3 scaling experiments</small></div>
  <div class="batch-stat"><strong>+6%</strong><small>Reported compute efficiency<br>Qwen3 MoE scaling experiments</small></div>
  <div class="batch-stat"><strong>Same budget</strong><small>Change the allocation<br>across optimizer updates</small></div>
</div>

<p class="batch-caption">The efficiency figures are reported in Figure 5. They are compute comparisons, not measured wall-clock speedups.</p>

* In this note
{:toc}

## 1. Look at both tokens and steps

A larger batch processes more tokens per update. The two views of Figure 1 tell a complementary story: the ordinary dynamic schedule improves final loss, while a **larger average batch reaches the reference loss in fewer optimizer updates at a similar token budget**. Compare iterations on the left with tokens on the right.

<section class="batch-card" id="batch-results">
  <div class="batch-card-header">
    <div><p class="batch-card-kicker">Figure 1 / Pre-training</p><h3>Larger batches: fewer updates to the same loss</h3></div>
    <span class="batch-tag batch-tag--approx">Approximate curves</span>
  </div>
  <div class="batch-controls">
    <label>Model<select data-result-model><option value="dense">Llama3-7B</option><option value="moe">Qwen3-A1B (MoE)</option></select></label>
  </div>
  <div class="batch-checks">
    <label><input type="checkbox" data-series-toggle value="static" checked> Static batch</label>
    <label><input type="checkbox" data-series-toggle value="dynamic" checked> Dynamic batch</label>
    <label><input type="checkbox" data-series-toggle value="large" checked> Dynamic, larger average batch</label>
  </div>
  <div class="batch-chart-grid">
    <div class="batch-result-panel">
      <h4>Optimizer iterations</h4>
      <div class="batch-chart" data-result-steps-chart><p class="batch-muted">Loading interactive chart...</p></div>
      <p class="batch-caption" data-result-steps-note></p>
    </div>
    <div class="batch-result-panel">
      <h4>Training tokens</h4>
      <div class="batch-chart" data-result-tokens-chart><p class="batch-muted">Loading interactive chart...</p></div>
      <p class="batch-caption" data-result-tokens-note></p>
    </div>
  </div>
  <p class="batch-caption">Source: <a href="https://arxiv.org/pdf/2607.27731v1#page=3">Figure 1</a>, FineWeb-Edu at a fixed 10<sup>20</sup>-FLOP budget. Curves are approximate reconstructions; iteration-reduction labels are taken from the paper.</p>
</section>

## 2. Let the learning-rate shape choose the batches

The answer is not always an increasing ramp. **Cosine** calls for a rise and fall. **WSD** calls for a rise followed by a plateau during cooldown. **Linear decay** gives a constant batch size. Change the peak learning rate below: the batch-size shape stays the same.

<section class="batch-card" id="batch-schedule">
  <div class="batch-card-header">
    <div><p class="batch-card-kicker">Table 2 / Schedule explorer</p><h3>Choose a learning-rate shape</h3></div>
    <span class="batch-tag batch-tag--exact">Analytic schedules</span>
  </div>
  <div class="batch-controls">
    <label>Learning-rate schedule<select data-schedule><option value="wsd">WSD: stable, then linear decay</option><option value="cosine">Cosine decay</option><option value="linear">Linear decay</option><option value="constant">Constant learning rate</option></select></label>
    <label>WSD cooldown: <output data-cooldown-value>10%</output><input data-cooldown type="range" min="5" max="50" step="1" value="10"></label>
    <label>Average batch (sequences)<select data-average><option value="256">256</option><option value="512">512</option><option value="1024" selected>1,024</option><option value="2048">2,048</option></select></label>
    <label>Peak-LR multiplier: <output data-peak-value>1.00x</output><input data-peak type="range" min="0.25" max="2" step="0.25" value="1"></label>
  </div>
  <div class="batch-chart-grid">
    <div class="batch-chart" data-lr-chart></div>
    <div class="batch-chart" data-batch-shape-chart></div>
  </div>
  <div class="batch-controls"><label>Inspect training progress: <output data-progress-value>50%</output><input data-progress type="range" min="0" max="100" step="0.1" value="50"></label></div>
  <dl class="batch-live-metrics">
    <div><dt>Learning rate here</dt><dd data-lr-value></dd></div>
    <div><dt>Ideal batch here</dt><dd data-batch-value></dd></div>
    <div><dt>Budget used here</dt><dd data-budget-value></dd></div>
  </dl>
  <div class="batch-chart" data-token-chart></div>
  <p class="batch-muted">Hover a plot or use the progress slider to align all three cursors: learning rate, batch size, and the token budget consumed at the same point in training.</p>
  <p class="batch-caption">Computed from <a href="https://arxiv.org/pdf/2607.27731v1#page=6">Table 2</a>, without warmup or integer-batch rounding. Fixed sequence length is assumed, so equal total sequences imply equal total tokens. The constant-LR curve has a singular endpoint and is displayed only up to 99.5% of training; cosine approaches zero. Real runs need feasible batch bounds and rounding.</p>
</section>

<details class="batch-detail batch-math" markdown="1">
<summary>The schedule formulas and why peak learning rate cancels</summary>

Fix the learning-rate schedule $\eta(t)$, the horizon $T$, and a total budget $K$. Define the remaining learning-rate area and its associated noise weight:

$$
S(t)=\int_t^T\eta(s)\,ds,
\qquad
w(t)=\frac{\eta(t)^2}{S(t)}.
$$

[Theorem 3](https://arxiv.org/pdf/2607.27731v1#page=5) gives

$$
B^*(t)
=K\frac{\sqrt{w(t)}}{\int_0^T\sqrt{w(s)}\,ds}
=\frac{K\eta(t)}{2\sqrt{S(0)S(t)}}.
$$

The normalization uses $\int_0^T\eta(t)/\sqrt{S(t)}\,dt=2\sqrt{S(0)}$, so $\int_0^T B^*(t)\,dt=K$. If $\eta(t)=a f(t)$ and $F(t)=\int_t^T f(s)\,ds$, then

$$
B^*(t)
=\frac{Kaf(t)}{2\sqrt{aF(0)\,aF(t)}}
=\frac{Kf(t)}{2\sqrt{F(0)F(t)}}.
$$

The peak multiplier $a$ cancels. That is why changing peak learning rate in the explorer changes the LR plot but leaves the batch allocation unchanged.

For the shapes in [Table 2](https://arxiv.org/pdf/2607.27731v1#page=6), set $x=t/T$, $\bar B=K/T$, and $\beta(x)=B^*(xT)/\bar B$:

$$
\begin{aligned}
\beta_{\mathrm{constant}}(x)&=\frac{1}{2\sqrt{1-x}},\\
\beta_{\mathrm{linear}}(x)&=1,\\
\beta_{\mathrm{cosine}}(x)&=
\frac{\cos^2(\pi x/2)}{\sqrt{1-x-\sin(\pi x)/\pi}}.
\end{aligned}
$$

For WSD, let $c$ be the fraction of training before linear cooldown begins:

$$
\beta_{\mathrm{WSD}}(x)=
\begin{cases}
\displaystyle\frac{1}{\sqrt{(1+c-2x)(1+c)}},&0\le x\le c,\\[6pt]
\displaystyle\frac{1}{\sqrt{1-c^2}},&c<x\le1.
\end{cases}
$$

Endpoint values are interpreted as limits. The derivation optimizes a convex-SGD loss characterization with fixed learning-rate schedule and horizon. Its usefulness for nonconvex neural networks and adaptive optimizers is tested empirically, rather than guaranteed by that theorem.

</details>

## 3. Spend the budget where it reduces noise most

In the characterization, noise from different updates carries different weight in the final loss. A constant batch spends equally on unequal contributions. Move the slider from **static** to **derived**, then keep going: exaggerating the allocation can give back the benefit.

<section class="batch-card" id="batch-allocation">
  <div class="batch-card-header">
    <div><p class="batch-card-kicker">Theorem 3 / Budget allocation</p><h3>Find the useful amount of scheduling</h3></div>
    <span class="batch-tag batch-tag--exact">Theoretical illustration</span>
  </div>
  <div class="batch-controls">
    <label>Learning-rate shape<select data-allocation-schedule><option value="wsd">WSD, 10% cooldown</option><option value="cosine">Cosine</option><option value="linear">Linear</option></select></label>
    <label>Allocation strength: <output data-allocation-power-value>0.00</output><input data-allocation-power type="range" min="0" max="2" step="0.05" value="0"></label>
  </div>
  <p class="batch-muted">0 = static &nbsp; / &nbsp; 1 = derived schedule &nbsp; / &nbsp; 2 = exaggerated schedule</p>
  <div class="batch-chart-grid">
    <div class="batch-chart" data-allocation-batches></div>
    <div class="batch-chart" data-allocation-cost></div>
  </div>
  <p class="batch-muted">Hover either plot to align both cursors at the same training progress: <output data-allocation-progress-value>50%</output>. Compare the batch allocated here with the noise contribution accumulated so far.</p>
  <p class="batch-insight" data-allocation-score></p>
  <p class="batch-caption">A numerical illustration of the batch-dependent noise term in <a href="https://arxiv.org/pdf/2607.27731v1#page=5">Equation 3.3</a>. Every allocation uses the same total budget. The right plot accumulates this term over training, normalized by the static schedule's full-run value; its endpoint matches the comparison above. The derived schedule minimizes the full-run total, not every intermediate cumulative value. This is not an observed training curve or a prediction of total LLM loss. For linear decay, the derived schedule is already static.</p>
</section>

<details class="batch-detail batch-math" markdown="1">
<summary>The allocation objective, a short proof, and what the slider changes</summary>

Up to a fixed positive factor, the batch-dependent part of the loss characterization is

$$
J[B]=\int_0^T\frac{w(t)}{B(t)}\,dt,
\qquad
\int_0^T B(t)\,dt=K,
\qquad
w(t)=\frac{\eta(t)^2}{S(t)}.
$$

The budget couples all updates: increasing one batch leaves fewer examples for others. Cauchy-Schwarz gives the best achievable value directly:

$$
\begin{aligned}
\left(\int_0^T\sqrt{w(t)}\,dt\right)^2
&=\left(\int_0^T
\sqrt{\frac{w(t)}{B(t)}}\sqrt{B(t)}\,dt\right)^2\\
&\le
\left(\int_0^T\frac{w(t)}{B(t)}\,dt\right)
\left(\int_0^T B(t)\,dt\right)\\
&=KJ[B].
\end{aligned}
$$

Equality holds when $B(t)$ is proportional to $\sqrt{w(t)}$. Normalizing to the budget gives $B^*$ and

$$
J[B^*]=\frac{1}{K}
\left(\int_0^T\sqrt{w(t)}\,dt\right)^2.
$$

The allocation-strength slider explores a budget-matched family around that solution:

$$
B_\alpha(t)
=K\frac{[B^*(t)]^\alpha}
{\int_0^T[B^*(s)]^\alpha\,ds}.
$$

At $\alpha=0$, this is the static schedule $B_{\mathrm{static}}=K/T$. At $\alpha=1$, it is the derived optimum. Larger values exaggerate the shape. Every choice still integrates to $K$.

The right-hand plot shows cumulative noise contribution at progress $x$:

$$
C_\alpha(x)=
\frac{\displaystyle\int_0^{xT}\frac{w(t)}{B_\alpha(t)}\,dt}
{\displaystyle\int_0^T\frac{w(t)}{B_{\mathrm{static}}}\,dt}.
$$

Its endpoint $C_\alpha(1)=J[B_\alpha]/J[B_{\mathrm{static}}]$ is the displayed noise-term ratio. The optimum minimizes the full-run term; it need not minimize its cumulative value at every intermediate time. These are calculations from the characterization, not measured LLM losses.

</details>

## 4. Scale the whole recipe

For a training horizon four times longer, the joint recipe halves both peak learning rate and weight decay. The normalized batch shape is unchanged; the budget sets its overall size. This combines batch scheduling with a transfer rule for the other hyperparameters.

<section class="batch-card" id="batch-scale">
  <div class="batch-card-header">
    <div><p class="batch-card-kicker">Section 4 / Scaling recipe</p><h3>Plan a longer training run</h3></div>
    <span class="batch-tag batch-tag--exact">Analytic transfer rule</span>
  </div>
  <div class="batch-controls">
    <label>Training-horizon multiplier: <output data-horizon-value>4x</output><input data-horizon type="range" min="1" max="16" step="1" value="4"></label>
  </div>
  <dl class="batch-live-metrics">
    <div><dt>Peak LR, starting from 0.01</dt><dd data-scale-rate></dd></div>
    <div><dt>Weight decay, starting from 0.1</dt><dd data-scale-decay></dd></div>
  </dl>
  <div class="batch-chart" data-scaling-chart></div>
  <p class="batch-caption">The 1 / square-root-of-horizon rule in <a href="https://arxiv.org/pdf/2607.27731v1#page=8">Equation 4.1</a>. The starting values are illustrative, not tuned recommendations. Figure 5's reported compute gains use joint scaling and the dynamic batch schedule; they are distinct from Figure 1's reductions in optimizer steps.</p>
</section>

<details class="batch-detail batch-math" markdown="1">
<summary>The horizon-transfer equations</summary>

Let $r=T/T_{\mathrm{ref}}$ compare a new horizon with a reference run. The joint rule in [Equation 4.1](https://arxiv.org/pdf/2607.27731v1#page=8) is

$$
\eta_{\mathrm{peak}}(T)
=\frac{\eta_{\mathrm{peak,ref}}}{\sqrt r},
\qquad
\kappa(T)=\frac{\kappa_{\mathrm{ref}}}{\sqrt r},
$$

where $\kappa$ is weight decay. At $r=4$, both multipliers are $1/2$. For a fixed normalized learning-rate shape, the batch schedule can be written

$$
B_T^*(t)=\frac{K}{T}\,\beta(t/T).
$$

The shape $\beta$ is unchanged, while $K/T$ sets its average size. Keeping total tokens fixed and keeping average batch fixed are different choices: the former changes the average as $T$ changes; the latter changes the total token budget. The explorer illustrates the transfer factors, rather than prescribing tuned starting values.

</details>

## 5. Check the downstream trade-offs

Better validation loss should also be checked against the tasks we care about. In Table 1, the dense model improves on all eight reported benchmarks. The MoE model improves on seven, with a small ARC-Challenge regression. Inspect the exact scores below.

<section class="batch-card" id="batch-benchmarks">
  <div class="batch-card-header">
    <div><p class="batch-card-kicker">Table 1 / Last-iterate evaluation</p><h3>Which benchmarks improve?</h3></div>
    <span class="batch-tag batch-tag--exact">Exact reported scores</span>
  </div>
  <div class="batch-controls">
    <label>Model<select data-benchmark-model><option value="dense">Llama3 (dense)</option><option value="moe">Qwen3 (MoE)</option></select></label>
  </div>
  <div class="batch-chart" data-benchmark-chart></div>
  <p class="batch-insight" data-benchmark-note></p>
  <p class="batch-caption">Exact values from <a href="https://arxiv.org/pdf/2607.27731v1#page=2">Table 1</a>, evaluated with lm-evaluation-harness v0.4. Differences are percentage points, not relative percentages. Hover or focus the chart to inspect individual tasks.</p>
</section>

<details class="batch-detail">
  <summary>Can changing the batch simply replace learning-rate decay?</summary>
  <p>Matching the learning-rate-to-batch ratio does not make two runs equivalent. Appendix C.5 compares schedules with the same ratio trajectory and total token budget: realizing the schedule through the learning rate gives lower final loss in every reported comparison.</p>
  <section class="batch-card" id="batch-ratio">
    <div class="batch-card-header">
      <div><p class="batch-card-kicker">Table 4 / A useful counterexample</p><h3>Same ratio, different final loss</h3></div>
      <span class="batch-tag batch-tag--exact">Exact reported losses</span>
    </div>
    <div class="batch-controls">
      <label>Base learning rate<select data-ratio-lr><option value="0.003">0.003</option><option value="0.001">0.001</option><option value="0.0003">0.0003</option></select></label>
      <label>Schedule shape<select data-ratio-schedule><option value="cosine">Cosine</option><option value="linear">Linear</option><option value="wsd">WSD</option></select></label>
    </div>
    <div class="batch-chart" data-ratio-chart></div>
    <p class="batch-insight" data-ratio-note></p>
    <p class="batch-caption">Source: <a href="https://arxiv.org/pdf/2607.27731v1#page=24">Table 4</a>. Lower loss is better. The paper's reported gaps can differ slightly from subtracting its rounded printed losses. This experiment tests interchangeability; it is separate from comparing the derived schedule against static batches.</p>
  </section>
</details>

## 6. What to try in a training run

Keep the comparison simple: **fix the model, sequence length, token budget, and learning-rate schedule**, then compare static batches with the derived allocation. Implement feasible effective batches with rounding and appropriate gradient accumulation. Check tokens, FLOPs, wall time, and downstream quality separately.

The benefit is not universal. It shrinks as the baseline batch grows, and the paper finds that it can disappear in purely low-precision training. Those are reasons to test the recipe in your own stack, not to assume a headline gain will transfer unchanged.

<p class="batch-insight"><strong>The takeaway:</strong> batch size is a schedule worth designing alongside learning rate. The useful question is where each token does the most work.</p>

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{li2026jointscaling,
  title         = {Towards joint scaling laws with optimal batch size schedules},
  author        = {Jiaxiang Li and Zhiqi Bu and Shiyun Xu},
  year          = {2026},
  eprint        = {2607.27731},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2607.27731},
  url           = {https://arxiv.org/abs/2607.27731}
}
```

<noscript><p class="batch-draft-note">Interactive figures require JavaScript. The surrounding explanations and source links remain available; please see the paper for the original figures and exact tables.</p></noscript>

</div>

<script src="{{ '/assets/js/batch-blog-toc.js' | relative_url }}" defer></script>
<script src="{{ '/assets/js/batch-size-blog.js' | relative_url }}" defer></script>
