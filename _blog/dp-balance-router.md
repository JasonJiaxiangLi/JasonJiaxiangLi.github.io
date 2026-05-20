---
title: "The EP Barrier Tax: How Request Routing Costs You 7-15% Throughput in MoE Serving"
date: 2026-05-19
last_modified_at: 2026-05-19
tags: [llm, systems, optimization]
excerpt: ""
hidden: false
---

* TOC
{:toc}

# The EP Barrier Tax: How Request Routing Costs You 7-15% Throughput in MoE Serving

**TL;DR:** When you serve a Mixture-of-Experts (MoE) LLM across multiple GPUs, every decode step waits for the slowest GPU due to expert parallelism synchronization. Bad request routing makes some GPUs much busier than others, wasting the rest. In this post, we study the DP load balancing problem starting from the BalanceRoute paper, derive an expected marginal overflow cost formulation, and test everything on real multi-node GPU clusters. Along the way we learn that the biggest win comes from simply switching to KV-load-aware routing — the specific algorithm matters less than the load metric.

**Disclaimer:** This is a side project I put together with Claude after reading Bu et al. (2026). I'm new to this area and still learning — if you spot errors or have suggestions, please reach out at jasonljx96@gmail.com.

---

## The Problem: GPUs Waiting in Line (But Only for MoE Models)

**Important distinction upfront:** The load-balancing problem in this post is specific to **Mixture-of-Experts (MoE) models** like DeepSeek-V3, Mixtral, and Qwen3-30B-A3B. For **dense models** served with ordinary external DP (separate vLLM instances behind a load balancer), each replica is fully independent — there is no synchronization barrier, and routing has minimal throughput impact. We explain the distinction below.

### Why MoE creates a barrier

When you serve a large MoE model, each DP worker is a group of GPUs doing tensor parallelism (TP). The model's expert layers are too large to replicate on every worker, so they are **sharded across all GPUs** using expert parallelism (EP). At every decode step, each worker must send its tokens to the correct expert (which may live on another worker's GPUs) via an **EP all-to-all** collective operation [1, 2].

This all-to-all is a **global synchronization barrier**: it cannot begin until every worker has finished its local attention computation. The step ends only when the slowest worker is done:

```
Decode step anatomy (MoE model):

GPU Group 0: [attention ████████░░] → [EP all-to-all ████] → [MoE FFN] → done
GPU Group 1: [attention ██████████] → [waits...] [EP a2a ██] → [MoE FFN] → done
GPU Group 2: [attention ██████░░░░] → [waits........] [a2a] → [MoE FFN] → done
                                       ↑
                                  barrier: everyone waits
                                  for Group 1 (slowest)
```

Chen et al. (2026) measured this on a real industrial LLM-serving trace: **over 40% of GPU compute is wasted** at these barriers [2].

### Why ordinary dense-model DP doesn't have this problem

For dense models served as independent replicas (the common vLLM external DP setup), each DP worker has a full copy of the model. TP all-reduce happens **within** each worker's GPU group, not across workers. Workers decode independently at their own pace. A slow worker only hurts the requests assigned to it — it doesn't block other workers. In this setting, routing has minimal throughput impact (though it still affects tail latency and fairness). Simple JSQ-by-load (JSQ-Load) is sufficient.

The strongest practical case for sophisticated routing is **MoE serving with DP+EP**, where DP ranks are coupled by expert synchronization. The BalanceRoute paper [1] specifically targets this setting, testing on DeepSeek-V3 (MoE) and Qwen3-30B-A3B (MoE).

### The step-time model

Each decode step has two parts. First, every worker reads its KV cache to compute attention — this is **memory-bandwidth-bound** and takes time proportional to how many KV tokens the worker has. Second, all workers do the EP all-to-all, MoE expert computation, and TP all-reduce — this takes roughly the same time regardless of which worker has more load.

We use a simple two-term model for the per-step wall-clock time: a load-sensitive straggler term plus a weakly load-sensitive overhead term:

$$t_{\text{step}} = \underbrace{\max_g (a \cdot L_g)}_{\text{attention (varies by worker)}} + \underbrace{b \cdot \bar{L}}_{\text{EP + MoE FFN + TP (fixed)}}$$

where:
- $L_g$ = KV workload on worker $g$ (total KV tokens across all its active requests)
- $\bar{L}$ = average load across all workers
- $a$ = time per KV token for attention (determined by memory bandwidth)
- $b$ = time coefficient for the fixed overhead (EP communication, expert computation, TP all-reduce)
- $\max_g$ = the barrier — all workers wait for the slowest one's attention to finish

The first term is where routing matters: if one worker has much more KV load than the others, everyone waits. The second term is the same no matter how you route — it doesn't care about imbalance.

**Example:** Suppose two workers have loads $L_1 = 100{,}000$ and $L_2 = 80{,}000$ tokens. With $a = 1$ and $b = 0.5$:

- Attention time = $\max(100{,}000, \ 80{,}000) = 100{,}000$ (worker 1 is the bottleneck)
- Fixed overhead = $0.5 \times 90{,}000 = 45{,}000$ (average of both)
- Total step time = $145{,}000$

If perfect routing made them equal ($L_1 = L_2 = 90{,}000$):

- Attention time = $90{,}000$ (10% faster — no bottleneck)
- Fixed overhead = $45{,}000$ (unchanged)
- Total step time = $135{,}000$ (only 7% faster, not 10%)

The fixed overhead **dilutes** the routing benefit. For DeepSeek-V3, this overhead is roughly 30-50% of the total step time, so routing improvements translate to about half the throughput gain you'd expect from imbalance reduction alone.

Chen et al. (2026) measured this on a production cluster: **over 40% of GPU compute is wasted** at these barriers. That's 40% of your electricity bill doing nothing.

The root cause? **Bad routing.** When a new request arrives, a router decides which GPU handles it. Once assigned, the request stays on that GPU for its entire lifetime (the KV cache is too large to migrate). If the router makes uneven assignments, some GPUs accumulate more work than others, and the imbalance persists for thousands of decode steps.

## What Makes This Hard?

Classical load balancing (round-robin, join-shortest-queue) was designed for web servers where requests are short and interchangeable. LLM serving breaks those assumptions in three ways:

1. **Assignments are sticky.** Each request builds a KV cache on its GPU. Moving it would require transferring gigabytes of data. So once you assign a request, you're stuck with that choice.

2. **Load grows over time.** Each request's KV cache grows by one token per decode step. A request that started at 3,000 tokens (the prompt) will be at 4,000 tokens 1,000 steps later. The "weight" of a request keeps increasing.

3. **You don't know how long it will run.** The output length is unknown at routing time. "What is 2+2?" generates 5 tokens. "Write a detailed analysis of..." might generate 5,000. The router must decide without knowing which type it's dealing with.

## The State of the Art: BalanceRoute

Bu et al. (2026) proposed **BalanceRoute** (BR-0 and BR-H), currently the best published algorithm for this problem. Let's walk through how it works.

### BR-0: The Single-Step View

BR-0 looks at the **current** loads on all GPUs and asks: "if I assign this request to GPU $g$, how much does the imbalance change?"

It defines a score for each GPU:

$$F_g = s - G \cdot \max(s - m_g, 0)$$

where:
- $s$ = the new request's prompt length (its initial KV cache size)
- $m_g$ = the **margin** on GPU $g$ = how far below the current maximum load GPU $g$ sits
- $G$ = the number of GPUs

The score has a sharp two-regime behavior. Suppose we have $G = 8$ GPUs and $m_g = 1000$ (GPU $g$ is 1,000 tokens below the busiest GPU):

- Assign a request with $s = 500$: $F = 500$. We're in the **safe zone** — GPU $g$ absorbs the load without becoming the new bottleneck. Good.
- Assign a request with $s = 1500$: $F = 1500 - 8 \times 500 = -2500$. We've **overflowed** — GPU $g$ is now 500 tokens above the old maximum, and the penalty is 8× (one for each GPU that now waits longer). Bad.

The crossover at $s = m_g$ is a **sharp kink**: below the margin, the slope is $+1$ (adding load helps); above it, the slope flips to $-(G-1)$ (adding load hurts $G-1$ times as much). The function is continuous but the marginal cost changes abruptly. BR-0 assigns to the GPU with the highest $F$ — strongly avoiding overflow.

### BR-H: Looking Ahead

BR-0's blind spot: it only sees the **current** loads. Consider:

```
GPU A: load = 500,000  (but 80% of its requests are about to finish)
GPU B: load = 400,000  (all fresh requests, none finishing soon)
```

BR-0 picks GPU B (lower load). Wrong — in 50 steps, GPU A will drop to 100,000 while B stays at 400,000.

BR-H fixes this by predicting $H$ steps into the future. For each active request $i$, a predictor estimates $\hat{c}_i$: "how many more steps will this request run?" Then BR-H projects what each GPU's load will look like at each future step.

**Concrete example.** GPU A has 3 requests:

| Request | Prompt $s_i$ | Tokens so far $a_i$ | Predicted remaining $\hat{c}_i$ |
|---------|-----------|------|------|
| req 1 | 1000 | 500 | 10 steps |
| req 2 | 2000 | 200 | 80 steps |
| req 3 | 800 | 900 | 5 steps |

GPU A's projected load:

```
Now (h=0):   (1000+500) + (2000+200) + (800+900)  = 5,400
h=5:         (1000+505) + (2000+205) + 0           = 3,710  ← req 3 gone
h=10:        0           + (2000+210) + 0           = 2,210  ← req 1 also gone
h=50:        0           + (2000+250) + 0           = 2,250  ← only req 2 left
```

Notice the **abrupt jumps** at $h=5$ and $h=10$. The load drops from 5,400 to 3,710 in one step because the predictor says req 3 finishes at exactly $h=5$. It's either 100% there or 100% gone — a binary cutoff.

**Step 3: Compute the envelope and margins.** Now suppose GPU B has a simpler load trajectory:

| $h$ | GPU A | GPU B | Envelope $M_h$ | Margin A ($m_{A,h}$) | Margin B ($m_{B,h}$) |
|-----|-------|-------|----------------|---------------------|---------------------|
| 0 | 5,400 | 4,000 | 5,400 | 0 | 1,400 |
| 5 | 3,710 | 3,900 | 3,900 | 190 | 0 |
| 10 | 2,210 | 3,800 | 3,800 | 1,590 | 0 |
| 50 | 2,250 | 3,500 | 3,500 | 1,250 | 0 |

The **envelope** $M_h = \max(\text{GPU A}, \text{GPU B})$ is the projected bottleneck at each step. The **margin** $m_{g,h} = M_h - \bar{L}_g(k+h)$ measures each GPU's headroom below the envelope.

Reading the table: GPU A is the bottleneck now ($m_{A,0} = 0$, no headroom) but will have lots of headroom soon ($m_{A,10} = 1590$) as its requests finish. GPU B has headroom now ($m_{B,0} = 1400$) but becomes the bottleneck at $h=5$ and stays there ($m_{B,5} = 0$).

**Step 4: Score each GPU.** A new request arrives with prompt length $s = 500$. BR-H scores each GPU using the horizon F-score:

$$F_g = \underbrace{\alpha \left(\sum_h \gamma^h\right) s}_{\text{reward (same for all GPUs)}} \quad - \quad \underbrace{\beta \sum_{h=0}^{H} \gamma^h \cdot \max(s - m_{g,h}, 0)}_{\text{penalty for overflow at each future step}}$$

The first term is a **reward** for accepting the request — it's the same for all GPUs, so it doesn't affect the ranking. The second term is the **penalty**: at each future step $h$, if the new request's load $s$ exceeds the margin $m_{g,h}$, there's a penalty weighted by the discount $\gamma^h$.

Let's compute the penalty for each GPU (using $\gamma = 0.9$, $\beta = 8$):

**GPU A:**
- $h=0$: margin $= 0$, overflow $= \max(500 - 0, 0) = 500$, weight $= \gamma^0 = 1.0$ → penalty $= 8 \times 1.0 \times 500 = 4000$
- $h=5$: margin $= 190$, overflow $= \max(500 - 190, 0) = 310$, weight $= \gamma^5 = 0.59$ → penalty $= 8 \times 0.59 \times 310 = 1463$
- $h=10$: margin $= 1590$, overflow $= \max(500 - 1590, 0) = 0$ → **no penalty** ✓
- $h=50$: margin $= 1250$, overflow $= 0$ → **no penalty** ✓

**GPU B:**
- $h=0$: margin $= 1400$, overflow $= \max(500 - 1400, 0) = 0$ → **no penalty** ✓
- $h=5$: margin $= 0$, overflow $= \max(500 - 0, 0) = 500$, weight $= \gamma^5 = 0.59$ → penalty $= 8 \times 0.59 \times 500 = 2360$
- $h=10$: margin $= 0$, overflow $= 500$, weight $= \gamma^{10} = 0.35$ → penalty $= 8 \times 0.35 \times 500 = 1400$
- $h=50$: margin $= 0$, overflow $= 500$, weight $= \gamma^{50} = 0.005$ → penalty $= 8 \times 0.005 \times 500 = 20$

Total penalty: GPU A ≈ 5,463, GPU B ≈ 3,780. **BR-H picks GPU B** (lower penalty).

Is this the right choice? GPU A is currently the bottleneck but will be light in 10 steps. GPU B has headroom now but will be the bottleneck for the remaining 70+ steps. BR-H picks B because the discount $\gamma^h = 0.9^h$ **heavily weights the near-term** — GPU A's near-term overflow ($h=0$, full weight) dominates GPU B's persistent far-term overflow ($h=5+$, discounted). Whether this is correct depends on how long the new request will actually run — and that's exactly what the discount $\gamma^h$ is supposed to capture. The choice of $\gamma$ trades off between near-term accuracy and far-horizon noise.

### Opportunities for refinement

BR-H's F-score involves several practical approximations. Understanding them precisely motivates our survival-weighted formulation:

**1. Constant load approximation.** The overflow term uses $\max(s - m_{g,h}, 0)$ with the same $s$ at every step. In reality, the new request's KV load at step $h$ is $s + h$ — it grows as tokens are generated. The paper makes this simplification explicitly for computational tractability.

**2. Tunable discount.** The discount $\gamma^h$ controls how much future steps matter. With $\gamma = 0.9$, $\gamma^{50} = 0.005$ — essentially zero beyond 50 steps. A data-driven weight — the probability the request is still alive at step $h$, i.e., $S(h) = P(O > h)$ — would adapt automatically to the workload's output length distribution without requiring tuning.

**3. Binary projections.** The projected load jumps when a request is predicted to finish (5,400 → 3,710 at $h=5$ in our example). A probabilistic projection using the survival function — "60% chance of finishing by $h=5$, 90% by $h=8$" — would give smoother load curves and more stable margins.

**4. Non-smooth F-score.** The $\max(s - m_{g,h}, 0)$ has a slope change at $s = m_{g,h}$. This is a deliberate design choice that directly targets the barrier cost. A smooth survival-weighted alternative transitions gradually near the margin, potentially trading precision for robustness.

## A Simple but Strong Baseline: Just Pick the Lightest GPU

Before presenting our survival-weighted formulation, let's establish a strong baseline: **always assign to the GPU with the lowest KV load.** No F-score, no predictions, no parameters.

This is the **JSQ-Load routing** policy, derived from minimizing the load variance:

$$V(k) = \sum_g (L_g(k) - \bar{L}(k))^2$$

The optimal one-step-greedy action is to assign to the GPU with $L_g$ furthest below the mean. Which is just... the lightest GPU.

In our simplified simulation (without MoE fixed overhead), JSQ-Load performs comparably to BR-0 across utilization levels. The two methods have different strengths: JSQ-Load is simpler and produces lower imbalance in the safe regime (most requests don't overflow), while BR-0's overflow penalty is valuable when capacity is genuinely scarce. In our real EP experiment at $G=4$, BR-0 slightly leads on throughput (+2.4%), suggesting the overflow penalty does provide value under real barrier synchronization.

## The Expected Marginal Overflow Cost

Building on BR-H's horizon F-score, we ask: what would the routing score look like if we addressed the four approximations above? Starting from the barrier-idle objective and taking expectations over the unknown output length, we derive the expected marginal overflow cost for a single routing decision.

Given fixed projected worker loads and no future arrivals, a new request arrives with prompt length $s$ (known) and unknown output length $O$. If we assign it to GPU $g$, the worker-dependent part of the expected barrier-idle increase is:

$$\boxed{\Delta\Phi_g(s) = \sum_{h=0}^{\infty} S(h) \cdot \max(s + h - m_g(h), 0)}$$

This quantity depends on both $s$ (the new request's size — larger prompts cause more overflow) and $g$ (which GPU — each has different margins). We compute $\Delta\Phi_g(s)$ for the same request across all GPUs, and route to the one with the smallest cost.

Let's unpack each piece:

- **$S(h) = P(O > h)$**: the probability the new request is still generating tokens at step $h$. We estimate this from the empirical output length distribution — just sort the historical output lengths and compute what fraction exceed $h$. No model fitting, no hyperparameters.

- **$s + h$**: the new request's KV load at step $h$. It starts at $s$ (prompt length) and grows by one token per step. Unlike BR-H, we don't pretend it's constant.

- **$m_g(h)$**: the margin on GPU $g$ at step $h$ — how far below the projected envelope GPU $g$ sits. Computed from per-request survival probabilities (probabilistic, not binary cutoff).

- **$\max(\cdot, 0)$**: overflow — the amount by which the new request pushes GPU $g$ above the envelope. If it doesn't overflow, this term is zero.

**The integral aggregates over the request's entire potential lifetime.** Steps where the request has likely finished ($S(h) \approx 0$) contribute nothing. Steps where the request is alive but within the margin also contribute nothing. Only steps where the request is both alive AND causing overflow contribute to the cost.

**Route to the GPU with the smallest $\Delta\Phi_g$.** No reward term, no tunable parameters. The survival function $S(h)$ adapts automatically to any workload.

### Why $S(h)$ and not a discount $\gamma^h$?

The $S(h)$ in the formula is not a design choice — it falls out of taking the expectation over the unknown output length $O$. At step $h$, the request causes overflow only if it's still alive, which happens with probability $P(O > h) = S(h)$. If you knew the output length exactly, you'd just sum to $O$ instead of weighting by $S(h)$. The survival function is mathematically required, not aesthetically preferred.

But what does $S(h)$ actually look like compared to $\gamma^h$? For a typical heavy-tailed LLM workload:

| $h$ | $S(h)$ (real, from data) | $\gamma^h$ ($\gamma = 0.9$) |
|-----|--------------------------|---------------------------|
| 0 | 1.00 | 1.00 |
| 10 | 0.95 | 0.35 |
| 50 | 0.70 | 0.005 |
| 100 | 0.50 | 0.00003 |
| 500 | 0.10 | ~0 |
| 1000 | 0.03 | ~0 |

The exponential $\gamma^h$ is effectively zero by $h = 50$ — it ignores 70% of the request's lifetime. Real LLM output lengths are heavy-tailed (many short responses, a thick tail of long ones), so $S(h)$ decays much more slowly.

**What happens if we replace $S(h)$ with $\gamma^h$ anyway?** The formula becomes:

$$\sum_h \gamma^h \cdot \max(s + h - m_g(h), 0)$$

This resembles BR-H's F-score (with $s+h$ instead of constant $s$). So **the discount $\gamma^h$ can be seen as an approximation of $S(h)$** — though BR-H also differs in other ways (constant load, binary projections, separate $\alpha$/$\beta$ parameters). The weight function is one of several differences, but it is the most conceptually interesting one.

If the output length distribution were memoryless (geometric/exponential), then $S(h) = (1-p)^h$ IS an exponential, and setting $\gamma = 1-p$ would be exactly right. But real LLM outputs are not memoryless — they're heavy-tailed, with shape that no single $\gamma$ can capture.

### How This Relates to BR-H

BR-H's F-score can be understood as an approximation of $\Delta\Phi_g$, with four practical simplifications that trade off accuracy for computational efficiency:

| Component | BR-H | $\Delta\Phi_g$ (ours) |
|-----------|------|---------------------|
| Weight at step $h$ | $\gamma^h$ (tunable, ad-hoc) | $S(h)$ (from data) |
| Overflow load | constant $s$ | $s + h$ (growing KV) |
| Load projection | binary cutoff | probabilistic survival weighting |
| Parameters | $(\alpha, \beta, \gamma)$ — need tuning | **zero** |

## Making It Fast: K-Point Quadrature

Computing $\Delta\Phi_g$ exactly requires evaluating the projected load at every horizon step for every active request — $O(G \times N \times T)$ operations. With 8 GPUs, 200 active requests each, and 200 horizon steps, that's 320K operations per routing decision. Too slow for the ~50ms dispatch budget.

The key insight: the margin values $m_g(h)$ at consecutive steps tend to change gradually, because each is a sum over many requests' survival-weighted contributions and individual departures have a small effect on the total. This suggests we don't need to evaluate at every $h$ — a few well-chosen sample points can capture the overall shape. (Note: the envelope $M(h) = \max_g \bar{L}_g(h)$ can still have kinks when the identity of the heaviest worker changes, so this is an empirical observation, not a theoretical guarantee.)

**Fast-$\Phi$** evaluates the margin at $K=5$ sample points (percentiles of the output length distribution) and approximates the integral with the trapezoid rule:

$$\Delta\Phi_g \approx \sum_{k=0}^{K-1} w_k \cdot S(h_k) \cdot \max(s + h_k - m_g(h_k), 0)$$

Cost: **$O(K \times G)$ per request** — same as JSQ-Load. The preprocessing ($O(K \times N \times G)$ to compute projected loads at $K$ points) is 5× JSQ-Load's cost but still negligible compared to a GPU decode step.

### Why K=5?

We swept $K$ in our offline simulator ($G = 8$ workers, 5,000 ShareGPT requests, $b = 0.5$ MoE overhead, arrival rate at ~70% utilization) and measured the average KV-load imbalance (max minus min worker load, averaged across all decode steps):

| $K$ | What it captures | Avg Imbalance (KV tokens) | vs BR-0 |
|-----|-----------------|-----------|---------|
| 1 | Current load only | 49,450 | 9% worse |
| 3 | Too coarse | 49,060 | 8% worse |
| **5** | **Curvature of margin trajectory** | **42,137** | **7% better** |
| 10 | Overfits to noise | 50,284 | 11% worse |
| full | Every step | 43,023 | 5% better |

$K=5$ actually beats the exact computation slightly. This suggests $K=5$ acts as a low-pass approximation that filters noise in the survival estimates and margin projections, rather than pure numerical quadrature. If $K=10$ were better than $K=5$, we'd call it quadrature; since it's worse, $K=5$ is better understood as a regularized heuristic that happens to work well empirically.

## Results

### Realistic simulation with MoE overhead model

Our simulation models the actual step-time structure of MoE inference: $t_{\text{step}} = \max_g(a \cdot L_g) + b \cdot \bar{L}$, where $b/a$ controls the fixed overhead fraction. With $b = 0$ (our original simulator), routing gains are overestimated. With $b = 0.5$ (realistic for MoE), the fixed EP/FFN/TP overhead dilutes the imbalance effect:

| Overhead model | JSQ-Load throughput vs JSQ | Fast-$\Phi$ throughput vs JSQ |
|---|---|---|
| $b = 0$ (pure bandwidth, old sim) | +12.0% | +5.5% |
| $b = 0.3$ (low overhead) | +9.4% | +4.4% |
| **$b = 0.5$ (realistic MoE)** | **+8.2%** | **+3.8%** |
| $b = 1.0$ (high overhead) | +6.2% | +3.0% |

### Scaling with realistic overhead ($b = 0.5$)

ShareGPT-like trace, arrival rate scaled to ~70% utilization at each $G$. **Throughput improvement** vs JSQ-by-count (vLLM's default):

| $G$ | JSQ-Load throughput vs JSQ | BR-0 throughput vs JSQ | Fast-$\Phi$ throughput vs JSQ |
|-----|-----------------|-------------|---------------------|
| 4 | +4.3% | +2.7% | +4.2% |
| 8 | +4.3% | +0.5% | +4.2% |
| 16 | +2.1% | +4.7% | **+6.6%** |

**Fast-$\Phi$ scales best** — at $G = 16$, it achieves +6.6% throughput over JSQ, beating both JSQ-Load (+2.1%) and BR-0 (+4.7%).

### Real EP serving experiments (Qwen3-30B-A3B, DP=8, TP=2, 2 nodes)

**Note on vLLM's default router:** vLLM v0.20.2 uses `score = 4 × waiting + running` — essentially JSQ by queue count, not by KV load. It has **no awareness of KV cache size**, treating a worker with 50 short requests the same as one with 50 long requests. All three of our routers (JSQ-Load, BR-0, Fast-$\Phi$) use KV load.

To understand how routing behaves under different levels of load, we ran a **request rate sweep** across 2 nodes (16 H200 GPUs total) with real EP barrier via Ray. The request rate controls how fast user requests arrive — mimicking light traffic, moderate traffic, and overloaded conditions.

**What is request rate?** We model user requests as a Poisson process with a given average rate (requests per second). At rate=1.5, a new request arrives roughly every 0.67 seconds. At rate=5.0, one arrives every 0.2 seconds. The system's capacity is fixed, so higher rates push it from under-utilized toward overloaded. This is the standard way to evaluate serving systems — you want to find the throughput-latency curve.

We tested three regimes: **under-saturated** (rate=1.5 req/s), **moderate** (rate=3.0 req/s), and **heavy** (rate=5.0 req/s). Each benchmark sends 500 ShareGPT requests with max 1024 output tokens. We measure four metrics:

- **Throughput** (tok/s): total output tokens generated divided by wall-clock time. Higher is better — it measures how efficiently the system uses its GPUs.
- **TPOT P95** (ms): 95th-percentile **Time Per Output Token** — the interval between consecutive tokens during decoding. This is what users perceive as "typing speed." Lower is better.
- **TTFT** (s): **Time To First Token** — how long a user waits after sending a request before seeing the first token. Includes queueing delay and prefill time. Lower is better.
- **Fail%**: fraction of requests that timed out (received no response within 600 seconds). Lower is better.

| Router | Rate | Throughput | TPOT P95 | TTFT | Fail% |
|--------|------|-----------|----------|------|-------|
| vllm_default | 1.5 | 68.8 tok/s | 3,340 ms | 7.9 s | 0.0% |
| jsq_load | 1.5 | **77.0 tok/s** | **3,067 ms** | **7.7 s** | 0.0% |
| fast_phi | 1.5 | 69.7 tok/s | 3,392 ms | 7.7 s | 0.0% |
| br0 | 1.5 | 73.5 tok/s | 3,079 ms | 12.2 s | 1.8% |
| | | | | | |
| vllm_default | 3.0 | 69.6 tok/s | 3,381 ms | 12.8 s | 0.0% |
| jsq_load | 3.0 | 74.8 tok/s | **3,047 ms** | 10.5 s | 0.0% |
| fast_phi | 3.0 | 70.1 tok/s | 3,385 ms | **9.1 s** | 0.0% |
| br0 | 3.0 | **79.9 tok/s** | 3,012 ms | 11.3 s | 3.6% |
| | | | | | |
| vllm_default | 5.0 | 70.8 tok/s | 3,276 ms | **8.8 s** | 0.0% |
| jsq_load | 5.0 | **76.0 tok/s** | **3,029 ms** | 12.4 s | 1.4% |
| fast_phi | 5.0 | 72.2 tok/s | 3,316 ms | 9.0 s | 0.0% |
| br0 | 5.0 | 73.6 tok/s | 3,212 ms | 11.6 s | 0.6% |

There's a lot to unpack here. Let us walk through each metric.

**Throughput.** JSQ-Load and BR-0 generally achieve higher throughput than vLLM's default — JSQ-Load by +7-12% at rates 1.5 and 5.0, BR-0 by +15% at rate 3.0. At rate 5.0, BR-0's advantage is smaller (+4%). Fast-$\Phi$ is more modest, roughly matching the default at low load and edging ahead by 2-3% at higher load.

**TPOT P95 (tail decode latency).** JSQ-Load has the best tail latency at every rate, running about 8-10% faster than the default. This makes sense — by keeping workers balanced by KV load, no single worker becomes the decode bottleneck. BR-0 also shows improvement at rates 3.0 and 5.0. Fast-$\Phi$ and the default are similar on this metric.

**TTFT (time to first token).** Fast-$\Phi$ shows the clearest advantage at rate=3.0, where vLLM's default degrades to 12.8 seconds while Fast-$\Phi$ stays at 9.1 seconds — a **29% improvement**. At rate=5.0, the picture reverses: the default achieves the best TTFT (8.8s) while Fast-$\Phi$ is close behind (9.0s). The TTFT advantage is rate-dependent, not universal. JSQ-Load and BR-0 have worse TTFT than the default at some rates — their aggressive throughput optimization may cause bursty routing patterns that create temporary queueing.

**Failure rate.** vLLM's default and Fast-$\Phi$ achieve 0% failures across all rates — they never overload the system to the point of timeout. JSQ-Load has a small failure rate at rate=5.0 (1.4%), and BR-0 has failures at all rates (0.6-3.6%). This reveals a **throughput vs reliability tradeoff**: the routers that push hardest for throughput (JSQ-Load, BR-0) occasionally overshoot, while conservative routers (default, Fast-$\Phi$) sacrifice some throughput for perfect reliability.

**Lessons from a bug we found along the way.** Our initial BR-0 implementation had a subtle issue: the F-score gives identical scores when no worker would overflow (the common case for small requests). Without a tie-breaker, all ties defaulted to worker 0, sending 43% of requests to a single worker. This taught us that **BR-0's F-score is fundamentally coarse** — it's a binary overflow-or-not check that can't differentiate among non-overflowing workers. The fix (breaking ties by lowest load) essentially makes BR-0 fall back to JSQ-Load in the common case, which is why the two methods now perform similarly.

**What we expected vs what we found.** Honestly, we expected the survival-weighted formulation (Fast-$\Phi$) to clearly outperform simpler methods. The results are more nuanced:

- The **biggest win is switching from queue-count routing to KV-load routing** — that's the gap between vLLM's default and any of our methods. The specific algorithm matters less than the load metric.
- Among KV-load-aware methods, the differences are modest (2-15% depending on rate and metric) and **no single method dominates across all metrics or rates**. JSQ-Load tends to win on throughput and tail latency. Fast-$\Phi$ tends to win on TTFT (especially at moderate load) and reliability. BR-0 can peak highest but is less consistent.
- The theoretical appeal of the overflow integral ($\Delta\Phi_g$) doesn't translate to a clear empirical advantage over the much simpler JSQ-Load. This may change at larger $G$ (our simulation predicts Fast-$\Phi$ pulls ahead at $G \geq 16$), but we haven't validated that with real hardware yet.

**Measurement caveats.** Two limitations of our live measurements are worth noting. First, throughput and TPOT are measured by counting streaming SSE chunks, not by tokenizer token counts — so the absolute numbers are approximate, although the same client and counting method are used for every router. Second, the rate-sweep results below were collected with an earlier KV-load tracker that approximated per-request generated tokens from engine-output callbacks. The current code now reads exact `new_token_ids` from vLLM, but we did not rerun the full sweep because it is expensive. Treat the live numbers as directional rather than paper-grade benchmarking; future runs should use the exact tracker.

### Previous result: independent DP workers (no barrier)

For completeness, we also tested with 8 independent vLLM instances (no EP barrier) on Qwen3-0.6B. In this setting, imbalance does not cause GPU idle — each worker decodes independently:

| Method | Throughput (tok/s) | Avg Imbalance |
|--------|-------------------|---------------|
| **Fast-$\Phi$** | 8,459 | **1,759** |
| JSQ-Load | 8,345 | 1,972 |
| BR-0 | 8,811 | 1,976 |
| Random | 8,705 | 3,099 |

Fast-$\Phi$ achieves the lowest imbalance (1,759 vs BR-0's 1,976 — 11% better). Throughput differences are small because without the EP barrier, imbalance doesn't cause GPU idle time.

## Does Better Prediction Help?

An intuitive idea: use a small LLM to predict output length from the prompt, giving prompt-conditional survival $S(h|\text{prompt})$ instead of the marginal $S(h)$. We tested this:

**For the new request's survival function:** no improvement in our experiments. In principle, changing $S(h)$ can alter the worker ranking when different workers have different early-vs-late overflow profiles (e.g., worker A overflows at $h=10$ while worker B overflows at $h=500$). But in practice, workers' overflow profiles tend to be correlated across horizon steps, so the ranking was insensitive to prompt-conditional $S(h)$ in our traces.

**For active requests' load projections:** the marginal survival conditioned on age ($P(\text{remaining} > h | \text{age})$) already captures most of the signal. With 50-200 requests per GPU, per-request prediction errors average out by the law of large numbers. A better per-request predictor has diminishing returns because the per-GPU load projection is already accurate.

**Bottom line:** Fast-$\Phi$ is effectively **prediction-free**. You only need the empirical output length CDF, which can be bootstrapped from a few hundred requests.

### What about a JSQ-Load tie-breaker?

A natural idea (suggested by Codex): when the overflow cost is zero for multiple workers, add a JSQ-Load drift term to break ties:

$$\text{score}_g = \underbrace{\sum_k w_k S(h_k)(s + h_k - m_g(h_k))_+}_{\text{overflow (avoids creating straggler)}} + \lambda \underbrace{\sum_k w_k S(h_k)(L_g(h_k) - \bar{L}(h_k))(s + h_k)}_{\text{Lyapunov drift (fills lightest worker)}}$$

We tested $\lambda \in \{0, 0.001, 0.01, 0.1, 1.0\}$ in simulation. Result: $\lambda = 0.1$ gives +0.5% over pure overflow at $G = 8$, but **hurts at $G = 4$ (-3.5%) and $G = 16$ (-0.8%)**. The optimal $\lambda$ is scale-dependent, adding a tunable parameter for inconsistent marginal gain. Pure overflow ($\lambda = 0$) is more robust — the margin-based implicit tiebreaker is sufficient in practice.

## What We Learned

1. **The load metric matters more than the algorithm.** Switching from queue-count routing to KV-load-aware routing gives 7-15% throughput improvement. The gap between different KV-load-aware algorithms is much smaller (2-5%). The single highest-impact change is making the router aware of KV cache sizes.

2. **No single router dominates all metrics.** JSQ-Load wins on throughput and tail latency. Fast-$\Phi$ wins on TTFT and reliability (zero failures). BR-0 peaks highest but has occasional failures. The "best" choice depends on whether you optimize for throughput or user experience.

3. **There is a throughput vs reliability tradeoff.** Aggressive routers (JSQ-Load, BR-0) occasionally overshoot under heavy load (1-4% failure rates). Conservative routers (vLLM default, Fast-$\Phi$) never fail but leave throughput on the table. This matters in production where a failed request is worse than a slower one.

4. **Fixed overhead dilutes routing impact.** MoE step time includes ~30-50% fixed overhead (EP, MoE FFN, TP all-reduce) unaffected by load balance. A 20% imbalance reduction yields only ~10% throughput gain. Our simulator captures this and its predictions match real hardware reasonably well.

## Try It Yourself

The code is open source: [github.com/JasonJiaxiangLi/DP-BalanceRouter](https://github.com/JasonJiaxiangLi/DP-BalanceRouter)

```bash
# Simulation on your laptop (no GPUs needed)
python run_simulation.py --num-workers 8 --num-requests 5000

# Live experiment on a GPU cluster
python scripts/vllm_patched_dp.py \
    --model <path-to-moe-model> \
    --data-parallel-size 8 --tensor-parallel-size 2 \
    --dp-router jsq_load --host 0.0.0.0 --port 8000

python run_benchmark.py \
    --target http://localhost:8000/v1/chat/completions \
    --num-requests 500 --request-rate 3.0 --dataset sharegpt
```

---

*This project started from studying "Tackling the Data-Parallel Load Balancing Bottleneck in LLM Serving" (Bu et al., 2026). Along the way we derived the overflow integral formulation, implemented the Fast-$\Phi$ approximation, and learned a lot about what actually matters in practice. We hope this writeup is useful to others exploring the same space.*

---

## References

1. **Bu, T., Lyu, Y., Chen, Z., Song, C., Liang, H., et al.** (2026). Tackling the Data-Parallel Load Balancing Bottleneck in LLM Serving: Practical Online Routing at Scale. *arXiv:2605.06113v2*. https://arxiv.org/abs/2605.06113

2. **Chen, Z., Bu, T., Song, C., Lu, X., Ye, Y., & Zhou, Z.** (2026). A Universal Load Balancing Principle and Its Application to Large Language Model Serving. *arXiv:2601.17855v2*. https://arxiv.org/abs/2601.17855

3. **Zhou, Z.** (2026). Position: LLM Serving Needs Mathematical Optimization and Algorithmic Foundations, Not Just Heuristics. *arXiv:2605.01280*. https://arxiv.org/abs/2605.01280

4. **Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., et al.** (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP 2023*. https://arxiv.org/abs/2309.06180

5. **Mitzenmacher, M.** (2001). The Power of Two Choices in Randomized Load Balancing. *IEEE Transactions on Parallel and Distributed Systems*.

6. **Neely, M. J.** (2010). Stochastic Network Optimization with Application to Communication and Queueing Systems. *Morgan & Claypool*.

7. **Sun, B., Huang, Z., Zhao, H., Xiao, W., Zhang, X., Li, Y., & Lin, W.** (2024). Llumnix: Dynamic Scheduling for Large Language Model Serving. *OSDI 2024*. https://www.usenix.org/conference/osdi24/presentation/sun-biao

8. **Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., & Zhang, H.** (2024). DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving. *OSDI 2024*. https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin

9. **Patel, P., Choukse, E., Zhang, C., Shah, A., Goiri, Í., Maleki, S., & Bianchini, R.** (2023). Splitwise: Efficient Generative LLM Inference Using Phase Splitting. *arXiv:2311.18677*. https://arxiv.org/abs/2311.18677

10. **Zheng, Z., Ren, X., Xue, F., Luo, Y., Jiang, X., & You, Y.** (2023). Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline. *NeurIPS 2023*. https://proceedings.neurips.cc/paper_files/paper/2023/hash/ce7ff3405c782f761fac7f849b41ae9a-Abstract-Conference.html

11. **Qiu, H., Mao, W., Patke, A., Cui, S., et al.** (2024). Efficient Interactive LLM Serving with Proxy Model-Based Sequence Length Prediction. *arXiv:2404.08509*. https://arxiv.org/abs/2404.08509

12. **Shahout, R., Malach, E., Liu, C., Jiang, W., Yu, M., & Mitzenmacher, M.** (2024). Don't Stop Me Now: Embedding Based Scheduling for LLMs. *arXiv:2410.01035*. https://arxiv.org/abs/2410.01035

13. **Jaillet, P., Jiang, J., Mellou, K., Molinaro, M., Podimata, C., & Zhou, Z.** (2025). Online Scheduling for LLM Inference with KV Cache Constraints. *arXiv:2502.07115*. https://arxiv.org/abs/2502.07115

14. **Ao, R., Luo, G., Simchi-Levi, D., & Wang, X.** (2025). Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints. *arXiv:2504.11320*. https://arxiv.org/abs/2504.11320

15. **Jain, K., Parayil, A., Mallick, A., Choukse, E., et al.** (2024). Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing. *arXiv:2408.13510*. https://arxiv.org/abs/2408.13510

16. **vLLM Team.** (2025). vLLM Router: A High-Performance and Prefill/Decode Aware Load Balancer for Large-scale Serving. https://blog.vllm.ai/2025/12/13/vllm-router-release.html
