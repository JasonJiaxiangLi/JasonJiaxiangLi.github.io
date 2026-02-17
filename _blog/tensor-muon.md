---
title: "Tensor descent: when and how to optimize in high dimensions?"
date: 2026-02-16
last_modified_at: 2026-02-16
tags: [math, optimization]
excerpt: ""
---

## Introduction

In <a href="#ref-muon" class="cite-ref" id="cite-muon">[1]</a>, the authors proposed a brand new optimizer different from Adam-like optimizers, named Muon (<a href="#fig-muon">Figure 1</a>).

<figure id="fig-muon" style="text-align: center;">
  <img src="/images/blog/muon.png" alt="Muon" width="350">
  <figcaption>Figure 1: Muon optimizer overview from <a href="#ref-muon">[1]</a>.</figcaption>
</figure>

A key step is the `NewtonSchulz5`, which is an approximation of turning each of the singular value of the input matrix to 1.

```python
def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
```

This NewtonSchulz is exactly applying the polynomial $p(x)=ax+bx^3+cx^5$ for five times on the input matrix signular values. The specific polynomial $(a, b, c) = (3.4445, -4.7750, 2.0315)$ looks as follows:

<figure id="fig-ns" style="text-align: center;">
  <img src="/images/blog/ns-polynomial.png" alt="NS" width="400">
  <figcaption>Figure 2: Applying the NS polynomial (From <a href="#ref-muon">[1]</a>).</figcaption>
</figure>

## Tensor Muon?

A sudden idea came to me is that: Since we have multi-head attention, it seems to make more sense if we deal with each of the head as separate matrices, e.g. if $Q\in\mathbb{R}^{d\times d}$ with $Q = [Q_1,...,Q_{\text{num_head}}]$, we might want separate each $Q_i\in\mathbb{R}^{d\times d/\text{num_head}}$ as if they are a separate matrix. This idea is also backed by the new work <a href="#ref-spectral" class="cite-ref" id="cite-spectral">[2]</a>, where they observe that separating the heads and conduct Muon separately seems to provide better performance.

A further thought becomes, essentially each head is playing a separate role as another dimension in the attention matrix, and we can stack each $Q_i$ in a new dimension to form a tensor $\mathcal{Q}\in \mathbb{R}^{d\times d/\text{num_head}\times \text{num_head}}$. How should I do Muon for this new tensor?

In <a href="#ref-tsvd" class="cite-ref" id="cite-tsvd">[3]</a>, the authors proposed a tSVD method for third-order tensors that fits exactly in my purpose. The key idea is to define a tensor-tensor product $\ast$ via block-circulant structure: given $\mathcal{A}\in\mathbb{R}^{n_1\times n_2\times n_3}$ and $\mathcal{B}\in\mathbb{R}^{n_2\times \ell\times n_3}$, one takes the FFT along the third mode of both tensors, multiplies the corresponding frontal faces in the Fourier domain, and applies an inverse FFT. Under this product, one can define tensor transpose ($\mathcal{A}^T$ transposes each face and reverses faces $2$ through $n_3$), an identity tensor, and orthogonality ($\mathcal{Q}^T\ast\mathcal{Q}=\mathcal{I}$). The tSVD then decomposes $\mathcal{A}=\mathcal{U}\ast\mathcal{S}\ast\mathcal{V}^T$, where $\mathcal{U},\mathcal{V}$ are orthogonal tensors and $\mathcal{S}$ is f-diagonal (each frontal face is diagonal). In practice, this amounts to computing a standard matrix SVD for each frontal face in the Fourier domain, costing $O(n_1 n_2 n_3\log n_3 + n_3\cdot\text{SVD}(n_1, n_2))$.

Then, I read TEON <a href="#ref-teon" class="cite-ref" id="cite-teon">[4]</a> which did a different thing than I described above: instead of stacking heads within a layer, they stack gradients of the same type (Q with Q, K with K, V with V) from $K$ consecutive layers into a tensor $\mathcal{G}\in\mathbb{R}^{m\times n\times K}$, apply mode-1 matricization to get $M_1(\mathcal{G})\in\mathbb{R}^{m\times nK}$, orthogonalize this matrix, and fold back to update each layer. The mode choice is theoretically motivated: mode-1 is preferred when the top right singular vectors are aligned across layers. In practice, $K=2$ (stacking two neighboring layers) yields the best trade-off between capturing cross-layer correlations and maintaining singular vector alignment.

### Comparing TEON and tSVD-Muon

Both TEON and the proposed tSVD-based approach stack layer gradients into a tensor, but they differ in how they orthogonalize. TEON flattens the tensor via matricization and orthogonalizes the resulting matrix, while tSVD-Muon works in the Fourier domain along the stacking dimension, applying per-face orthogonalization and coupling the faces through the FFT.

<div class="algo-container">
<div class="algo-box">
<div class="algo-title">Algorithm 1: TEON (mode-1) [4]</div>
<div class="algo-io"><strong>Input:</strong> Parameters $\{W^{(k)}\}_{k=1}^N$, lr $\eta$, momentum $\mu$, group size $K$</div>
<ol>
<li>Compute gradients $\{G_t^{(k)}\}$ for each layer</li>
<li><strong>for</strong> each group of $K$ layers <strong>do</strong></li>
<li>&emsp;Stack: $\mathcal{G}_t\in\mathbb{R}^{m\times n\times K}$, $\mathcal{G}_t[:,:,k] = G_t^{(k)}$</li>
<li>&emsp;Momentum: $\mathcal{M}_t = \mu\,\mathcal{M}_{t-1} + \mathcal{G}_t$</li>
<li>&emsp;Matricize: $Z_t = M_1(\mathcal{M}_t)\in\mathbb{R}^{m\times nK}$</li>
<li>&emsp;Orthogonalize: $Q_t = \text{Ortho}(Z_t)$</li>
<li>&emsp;Fold back: $\mathcal{O}_t = M_1^{-1}(Q_t)$</li>
<li>&emsp;<strong>for</strong> $k=1,\dots,K$: $W^{(k)} \leftarrow W^{(k)} - \eta\sqrt{m/n}\;\mathcal{O}_t[:,:,k]$</li>
<li><strong>end for</strong></li>
</ol>
<div class="algo-cost"><strong>Cost:</strong> One $\text{Ortho}(\cdot)$ call on $\mathbb{R}^{m\times nK}$ per group.</div>
</div>
<div class="algo-box">
<div class="algo-title">Algorithm 2: tSVD-Muon (proposed)</div>
<div class="algo-io"><strong>Input:</strong> Parameters $\{W^{(k)}\}_{k=1}^N$, lr $\eta$, momentum $\mu$, group size $K$</div>
<ol>
<li>Compute gradients $\{G_t^{(k)}\}$ for each layer</li>
<li><strong>for</strong> each group of $K$ layers <strong>do</strong></li>
<li>&emsp;Stack: $\mathcal{G}_t\in\mathbb{R}^{m\times n\times K}$, $\mathcal{G}_t[:,:,k] = G_t^{(k)}$</li>
<li>&emsp;Momentum: $\mathcal{M}_t = \mu\,\mathcal{M}_{t-1} + \mathcal{G}_t$</li>
<li>&emsp;FFT along mode 3: $\tilde{\mathcal{M}}_t = \text{FFT}_3(\mathcal{M}_t)$</li>
<li>&emsp;<strong>for</strong> $k=1,\dots,K$: $\tilde{\mathcal{O}}_t[:,:,k] = \text{Ortho}(\tilde{\mathcal{M}}_t[:,:,k])$</li>
<li>&emsp;Inverse FFT: $\mathcal{O}_t = \text{IFFT}_3(\tilde{\mathcal{O}}_t)$</li>
<li>&emsp;<strong>for</strong> $k=1,\dots,K$: $W^{(k)} \leftarrow W^{(k)} - \eta\sqrt{m/n}\;\mathcal{O}_t[:,:,k]$</li>
<li><strong>end for</strong></li>
</ol>
<div class="algo-cost"><strong>Cost:</strong> $K$ calls of $\text{Ortho}(\cdot)$ on $\mathbb{R}^{m\times n}$ + two FFTs of size $K$.</div>
</div>
</div>

The key difference: TEON orthogonalizes one large $m\times nK$ matrix, coupling layers through column-concatenation. tSVD-Muon orthogonalizes $K$ separate $m\times n$ matrices in the Fourier domain, where the FFT/IFFT provides a different form of cross-layer coupling via circulant structure. When $K$ is small (e.g., 2), the FFT reduces to simple sums and differences of the faces, making tSVD-Muon easy to implement and interpret.

I didn't have time to compare the tSVD-Muon with TEON in detail. Below I only incude a preliminary comparison of the proposed tSVD-Muon to the SOTA NorMuon implementation in the <a href="https://github.com/KellerJordan/modded-nanogpt">modded-nanogpt</a> repo (I basically just made minimum changes over the NorMuon implementation in [this file](https://github.com/KellerJordan/modded-nanogpt/blob/e22a34bb076cba691977c5da04e490938ff2efbe/train_gpt.py#L329)). The result is presented in <a href="#fig-res">Figure 3</a>. Note that I did a small sweep over the learning rate for tSVD Muon.

<figure id="fig-res" style="text-align: center;">
  <img src="/images/blog/tsvd-Muon.png" alt="MuonRes" width="350">
  <figcaption>Figure 3: Result of tSVD Muon on GPT 120M model.</figcaption>
</figure>

The next questions are, 1. when is tensorized Muon better than the original Muon? 2. For big models, do we see an even bigger gain in performance or it get worse? To be continued...

## References

<ol class="references-list">
  <li id="ref-muon">K. Jordan et al., <em>Muon: An optimizer for hidden layers in neural networks</em>, <a href="https://kellerjordan.github.io/posts/muon/">webpost</a>, 2024. <a href="#cite-muon" class="cite-backref" title="Jump back to citation">&uarr;</a></li>
  <li id="ref-spectral">T. Xie et al., <em>Controlled LLM Training on Spectral Sphere</em>, <a href="https://arxiv.org/abs/2601">arXiv:2601</a>, 2026. <a href="#cite-spectral" class="cite-backref" title="Jump back to citation">&uarr;</a></li> 
  <li id="ref-tsvd">M. E. Kilmer, C. D. Martin, and L. Perrone, <em>A Third-Order Generalization of the Matrix SVD as a Product of Third-Order Tensors</em>, Tufts University, Department of Computer Science, Tech. Rep. TR-2008-4, 2008. <a href="#cite-tsvd" class="cite-backref" title="Jump back to citation">&uarr;</a></li>
  <li id="ref-teon">R. Zhang et al., <em>TEON: Tensorized Orthonormalization Beyond Layer-Wise Muon for Large Language Model Pre-Training</em>, <a href="https://arxiv.org/abs/2601.23261">arXiv:2601.23261</a>, 2026. <a href="#cite-teon" class="cite-backref" title="Jump back to citation">&uarr;</a></li>
</ol>

If you find this post useful, you can cite it as:

```bibtex
@misc{li2026tensor,
  author = {Li, Jiaxiang},
  title  = {Tensor descent: when and how to optimize in high dimensions?},
  year   = {2026},
  url    = {https://jasonjiaxiangli.github.io/blog/tensor-muon/},
  note   = {Blog post}
}
```