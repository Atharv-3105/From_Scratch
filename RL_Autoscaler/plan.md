# RL Autoscaler: Implementation Plan (from scratch, free, Kubernetes-integrated)

> **This file is written so that any LLM (or person) can read it cold and understand what is being built, why, how, and in what order.**
> Nothing here is implemented yet. Every number marked *(estimate)* or *(verify)* must be checked during the relevant brick.

---

## 0. How an LLM collaborator should use this document

1. **Read all of it before answering anything.** Sections 1-4 are the "why/what"; 5-13 are the specification; 14-15 are the plan; 16-19 are risks, open decisions, environment facts and references.
2. **Working style requested by the owner (Atharva):**
   - **Brick-by-brick.** Work on one brick (Section 15) at a time. Do not jump ahead or hand over the whole repo at once.
   - **The LLM writes the code and gives it to the owner in chat; the owner types it into the project himself.** This is deliberate: he learns by reading, understanding and typing each piece. So for every brick, deliver **complete, runnable, annotated code** (with the target file path, ready to copy) **plus** a plain-language walk-through of the logic and why each design choice was made. **Do not create, edit or run anything inside the project folder yourself** unless he explicitly asks you to.
   - **Verify before handing over.** Test the code you give him in a scratch location *outside* the project (for example a temp folder) so what you hand over is known to run. Say what you verified and how; if you could not verify something, say so.
   - **Explain, do not just paste.** For each block: what it does, why it is written that way, common mistakes, and how to check it. Prefer several small blocks (one function or class at a time) over one giant file, so he can follow along and type it in with understanding.
   - **Test first.** Give the test for a brick *before* the implementation. He runs it and watches it fail, then types in the implementation and watches it pass. Each brick's "Done when" lists the acceptance tests.
   - **Support his own edits.** If he changes your code or writes his own version, review it, explain the differences, and debug from his output instead of replacing it.
   - **Be honest about results.** If the RL agent does not beat a baseline, that is a valid result to report. Never tune only the RL side; baselines get the same tuning budget.
   - **State uncertainty.** Distinguish what was verified (ran it) from what is assumed. Mark estimates.
   - **Ask before** installing software, downloading large files, or doing anything outside the project folder.
   - **Everything must stay free**: no paid cloud, no paid APIs. Local machine + free Colab/Kaggle only if ever needed.
3. **When the owner asks "what next?"** Point to the first unchecked brick in Section 15 and restate its Goal / Done-when.

---

## 1. Aim and motivation (the reason this project exists)

**Owner:** Atharva, an AI/ML graduate who has implemented several RL algorithms from scratch (PPO, DQN, A3C, SAC) but only on Gym-style environments.

**Why this project:**
1. **Test whether RL is feasible for a real-world systems problem**, not a toy Gym task. Owner's view: RL is under-explored for real-world use, and he wants evidence, not opinion.
2. **Portfolio piece for LinkedIn / X** to gain traction and to signal to AI startups and companies that he can (a) implement RL from scratch, (b) engineer a system around it, (c) evaluate honestly against strong baselines, and (d) integrate with real infrastructure (Kubernetes).
3. **Learn Kubernetes** hands-on (Pods, Deployments, ReplicaSets, the scale subresource, kubectl, kind, RBAC).
4. It should be an **implementation / experiment project, not a research paper**: it should *work end-to-end* and *visibly show the RL agent learning*, while still comparing against real baselines.

**What "success" looks like to the owner:**
- A trained PPO autoscaling policy that runs behind a clean `Autoscaler` interface and **actually scales Deployments on a local Kubernetes (kind) cluster**, guarded by a safety layer.
- Compelling visuals of **what the agent learned and how its behaviour changed during training** (Section 11).
- A fair, reproducible comparison against published and standard autoscaling policies on **real production traces**, including an honest account of where RL wins and where it does not.
- A clear README + short demo GIF/video + write-up suitable for a LinkedIn/X post.

**What it is not:** a novel research contribution, a production-ready autoscaler, or a multi-cluster system. Say "prototype" in all public material.

---

## 2. Constraints

| Constraint | Detail |
|---|---|
| Cost | Free only. Local machine; free-tier Colab/Kaggle GPU only if truly needed (it should **not** be: the model is tiny, the simulator is CPU-bound). |
| Implementation | **From scratch**: simulator, baselines, PPO, metrics, feature pipeline, controller. Allowed libraries: NumPy, pandas, PyTorch (autograd/nn only), matplotlib, pytest, the official `kubernetes` Python client. **Not allowed:** Stable-Baselines3, RLlib, Gym/Gymnasium as the environment framework (a Gym-*like* API written by hand is fine), off-the-shelf autoscaling libraries. |
| Time | Target 2-3 days of focused work; realistically 4-5 evenings. Scope-cut order is in Section 16. |
| Hardware | Windows 11, Ryzen 5 5500U (6C/12T), 15 GB RAM, **no local GPU**. Docker Desktop with about 12 CPUs / 8 GB allocated. |
| Data | Public Azure Functions traces (CC-BY). Must cite the paper. |

---

## 3. The problem in plain language, and glossary

**Problem.** A service gets requests. You keep some number of instances (pods/containers) running and ready. If you keep too few, a new request must wait while a fresh instance starts (a **cold start**), which hurts latency. If you keep too many, you pay for **idle** instances. **Autoscaling** = choosing how many instances to have ready for the next minute, ahead of time.

**What the agent does (each minute, for each service/function):**
1. Observe a **state** (recent traffic, trend, time of day, current pool size, ...).
2. Choose an **action**: how much spare capacity to keep, expressed relative to recent demand (a *headroom* choice).
3. Real traffic for that minute is revealed (from the trace).
4. Receive a **reward**: minus (cold-start penalty + idle cost). PPO learns which headroom to choose in which situation.

**The baselines are NOT actions.** They are competing *policies* run on identical traffic so the comparison is fair. (A "selector agent that picks among baselines" is a valid but different design; it is an optional later extension, not the main project.)

**Glossary**
- **Instance / pod / replica:** one running copy of a service. In the simulator these are abstract; on Kubernetes they are Pods of a Deployment.
- **Function:** one serverless function in the Azure trace; we treat each as one "service".
- **Demand D:** number of instances that must be busy in a minute to serve that minute's invocations (derived in Section 7).
- **Pool P:** number of instances kept warm (the agent's decision).
- **Cold start:** demand that exceeds the pool, forcing on-demand instance starts.
- **Idle:** pool that exceeds demand (wasted).
- **Keep-alive:** keep an instance warm for K minutes after last use (AWS about 10 min, Azure about 20 min, per Shahrad et al.).
- **HPA:** Kubernetes Horizontal Pod Autoscaler, a reactive rule scaling on a utilisation target.
- **Headroom factor:** multiplier applied to a recent-demand estimate to set the pool.
- **Pareto curve:** cost vs cold-start trade-off; better policies sit lower-left.

---

## 4. Scope, non-goals, success criteria

**In scope:** trace loader; simulator; five baselines; PPO agent; evaluation harness; learning visualisations; `Autoscaler` interface + safety wrapper; live Kubernetes (kind) backend with trace replay; README and demo.

**Non-goals:** production hardening; multi-metric autoscaling (CPU+memory+queue); vertical scaling; node autoscaling; GPU training; any paid service.

**Success criteria (all must be honestly reported, pass or fail):**
- **S1 (simulator validity):** on the same trace, the reimplemented hybrid-histogram policy shows fewer cold starts than fixed 10-minute keep-alive at comparable memory, directionally matching Shahrad et al. If not, debug the simulator *before* any RL.
- **S2 (agent learns):** training curves improve and the checkpoint replays visibly change behaviour (Section 11).
- **S3 (fair comparison):** agent and baselines are tuned with the same budget on the same data; results are on held-out days and held-out functions, with confidence intervals over seeds.
- **S4 (where it wins):** results are reported per function class (predictable / bursty / rare / hot). It is acceptable that RL wins only on some classes, or on none.
- **S5 (integration):** the live demo runs end-to-end on kind: metrics -> features -> policy -> safety wrapper -> `kubectl`-equivalent scale action, with a logged reason for every decision.

---

## 5. Architecture

```
                      +-------------------+
   Azure traces --->  |  traces/          |  loader, demand builder, splits
                      +---------+---------+
                                |
                                v
   +-----------------+   +------+-------+   +--------------------+
   | baselines/      |   | sim/         |   | rl/                |
   | fixed, hybrid,  |<->| WarmPoolEnv  |<->| PPO, policy/value  |
   | HPA, seasonal,  |   | metrics      |   | training, ckpts    |
   | oracle          |   +------+-------+   +---------+----------+
   +--------+--------+          |                     |
            |                   v                     |
            |            +------+-------+             |
            +----------> | features/    | <-----------+     ONE shared FeatureBuilder
            |            +------+-------+                   (train/serve parity)
            v                   |
   +--------+-------------------+-------------------+
   | scaler/  Autoscaler interface + SafetyWrapper  |
   +--------+---------------------------+-----------+
            |                           |
            v                           v
   +--------+--------+         +--------+---------------------------+
   | Sim backend     |         | live/  Kubernetes backend (kind)   |
   | (train/benchmark)|        | replayer -> Deployments -> patch   |
   +-----------------+         | scale via K8s API                  |
                               +------------------------------------+
   viz/  reads logs + checkpoints from sim and live and renders figures
```

**Repo layout**
```
RL_Autoscaler/
  plan.md
  README.md
  requirements.txt
  configs/            # yaml/dataclass configs (seeds, splits, hyperparameters)
  data/               # raw/ (gitignored), processed/
  traces/             # loader.py, demand.py, splits.py
  features/           # builder.py (single source of truth for state features)
  sim/                # env.py (single function), vec_env.py, metrics.py
  baselines/          # fixed.py, hybrid.py, hpa.py, seasonal.py, oracle.py, tune.py
  rl/                 # ppo.py, networks.py, normalize.py, train.py, evaluate.py
  scaler/             # interface.py, safety.py, policy_adapter.py
  live/               # kind-config.yaml, worker/ (Dockerfile, app), replayer.py, controller.py
  viz/                # training.py, replay.py, heatmap.py, pareto.py, classes.py, fleet.py, importance.py
  reports/            # figures/, tables/, logs/
  tests/
```

**Critical interface (the "genuine integration" seam)**
```python
class Autoscaler(Protocol):
    def reset(self) -> None: ...
    def decide(self, obs: FunctionObs) -> Decision: ...   # Decision(desired_instances:int, reason:str, meta:dict)

class Backend(Protocol):
    def observe(self, function_id) -> FunctionObs: ...    # metrics window + current pool
    def apply(self, function_id, desired_instances: int) -> None: ...
```
`SimBackend` replays traces; `K8sBackend` reads live metrics and patches Deployment scale. **Both feed the same `FeatureBuilder`**, and a test asserts identical features for identical history (prevents train/serve skew).

---

## 6. Data

**Source:** Azure Functions Trace 2019 (Shahrad et al., USENIX ATC 2020), CC-BY. **Cite the paper** in README and any post.
- Repo: `github.com/Azure/AzurePublicDataset` (file `AzureFunctionsDataset2019.md`).
- Download URL reported in search results *(verify before use)*: `https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz`
- Size not confirmed: **check before downloading** and use a stratified subset if large.

**Files (per the dataset docs)**
- `invocations_per_function_md.anon.d01..d14.csv` (14 days): `HashOwner, HashApp, HashFunction, Trigger`, then **1440 columns = invocations per minute**.
- `function_durations_percentiles.anon.d01..d14.csv`: `Average` (ms), `Count`, `Minimum`, `Maximum`, percentiles (0,1,25,50,75,99,100). **Execution time excludes cold-start time.**
- `app_memory_percentiles.anon.d01..d12.csv` (last 2 days missing): `AverageAllocatedMb` and percentiles. Memory is per **app**, not per function.
- IDs are HMAC-hashed; consistent across files, so files can be joined on `(HashOwner, HashApp, HashFunction)` (memory on `(HashOwner, HashApp)`).

**Workload facts to reproduce as a sanity check (from the paper, *verify on our load*):** most functions are invoked rarely (about 45% of apps average 1 invocation/hour or less); the top ~19% of apps produce ~99.6% of invocations; only ~20% of apps have highly regular (CV about 0) inter-arrival times.

**Known caveats (must appear in README "Limitations")**
- Minute-binned: sub-minute bursts and inter-arrival times are invisible.
- Unknown sampling rate per app; counts and duration logs can diverge slightly.
- Durations exclude cold-start; average duration understates peaks.
- Memory missing for last 2 days; some min/max duration values missing.
- Data is from July 2019; workloads differ today.

**Function classes (used for stratified sampling and per-class reporting)**
- `rare`: mean rate < 1 invocation/hour
- `hot`: top decile by total invocations
- `predictable`: high day-over-day correlation of per-minute demand *or* low inter-arrival CV (define precisely in Brick B2 and freeze it)
- `bursty`: high inter-arrival CV / high peak-to-mean
Freeze the class definitions and thresholds in config; never change them after seeing test results.

**Splits (freeze in Brick B4)**
- Days 1-10 train, 11-12 validation, 13-14 test (test is touched only for final reporting).
- Functions: 70/15/15 split **by function**, stratified by class. Report both "held-out days" and "held-out functions".
- Seeded, saved to disk, with leakage tests.

---

## 7. Simulator specification

Discrete time, **1 step = 1 minute**. For function *f*, minute *t*:

```
n[f,t]  = invocations in that minute (from trace)
d[f]    = average duration in seconds (per-day average if available, else global average)
c       = concurrency per instance (default 1; sensitivity check with c in {1, 4})
mem[f]  = app memory in MB (median if missing)

D[f,t]  = 0                                   if n == 0
        = max(1, ceil(n * d / (60 * c)))      otherwise            # instances that must be busy
```

**Decision timing (no lookahead):** at the start of minute *t* the agent picks pool `P[f,t]` using **only information up to minute t-1**. Scale-out decided now is ready at the start of minute *t* (this is how proactive/pre-warming is modelled). Scale-in is immediate.

```
cold[f,t] = max(0, D - P)          # instances started on demand (cold starts)
idle[f,t] = max(0, P - D)          # warm but unused instance-minutes
idle_GBmin[f,t] = idle * mem[f] / 1024
start[f,t] = max(0, P[f,t] - P[f,t-1])    # optional start/churn cost (see below)
```

**Reference demand:** `R[f,t] = max(D[f, t-5 .. t-1])` (spike-aware, last 5 minutes).

**Action set (6 discrete actions; scale-free across functions of very different size):**

| idx | pool P |
|---|---|
| 0 | 0 (scale to zero) |
| 1 | 1 (keep one warm; essential for rare functions where R is often 0) |
| 2 | ceil(0.5 * R) |
| 3 | ceil(1.0 * R) |
| 4 | ceil(1.5 * R) |
| 5 | ceil(2.0 * R) |

Clamp `P` to `[0, P_max[f]]` where `P_max[f]` is a documented cap (e.g., 2x the maximum training demand, min 1).

**Reward (per step, in "instance-minutes"):**
```
r = -( rho * cold + idle + kappa * start ) / N[f]
N[f] = max(1, mean of positive D over the training days)       # function-scale normaliser
```
- `rho` = price of one cold start expressed in idle-instance-minutes. Sweep `rho in {3, 10, 30}` (and maybe {1, 100}); each value gives one point on the Pareto curve.
- `kappa` = optional churn/start cost (default small, e.g., 0.1; ablate with 0). It also makes the problem genuinely sequential.
- Also log **memory-weighted** idle (`idle_GBmin`) as a secondary metric to mirror the paper's "wasted memory time".

**Design insight to keep in mind (important):** demand is *exogenous* (the agent cannot change traffic). Without a start/churn cost and without `pool` in the state, every minute is an independent contextual bandit. We include `pool` in the state and a small `kappa` so sequential structure is real, and we **ablate with gamma = 0** to test whether sequential reasoning actually helps. Report this honestly.

**Episode:** one function-day (1440 steps), fixed horizon. State features are pre-computable per function-day because they depend only on demand history (exogenous) plus the agent's previous pool.

---

## 8. Baselines (all reimplemented from scratch, expressed as pool rules)

All baselines see the same information as the agent (history up to t-1) except the oracle.

1. **Fixed keep-alive (K minutes):** `P[t] = max(D[t-K .. t-1])` (instances used in the last K minutes stay warm). K in {1, 5, 10, 20, 60}. **K=10 is the AWS default; K=20 the Azure default.** Also "never unload" (K = infinity).
2. **Hybrid histogram (Microsoft, ATC'20):** per-function histogram of idle-time durations (1-minute bins, up to 4 hours). If the histogram is "representative" (low CV), set a function-specific **pre-warm time** (low percentile of idle time) and **keep-alive** (high percentile); otherwise fall back to a fixed keep-alive. Adapted to our pool model: after demand ends, pool drops to 0 until the pre-warm time, is 1 (or last peak) through the keep-alive window, and equals recent peak demand while active. **Document every deviation from the paper**; read paper section 5 during Brick B10 *(verify exact percentiles/thresholds there)*.
3. **HPA-style reactive:** Kubernetes formula `desired = ceil(current * currentUtilisation / targetUtilisation)` with a tolerance (default 0.1), scale-down **stabilisation window** (default 5 min: take the max recommendation in the window), `min/max` bounds. Utilisation = `D[t-1] / P[t-1]`. Tune `target` in e.g. {0.4 ... 0.9}. Note real HPA cannot scale to zero by default (min replicas 1); report both variants.
4. **Seasonal predictive:** forecast `Dhat[t] = max(D over [t-1440-w, t-1440+w])` (same window yesterday), `P[t] = ceil(m * max(Dhat[t], D[t-1]))` with margin `m` tuned. Falls back to reactive on day 1.
5. **Oracle:** `P[t] = D[t]` (zero cold, zero idle). Reported only as a **cost floor / sanity bound**, not a competitor.

**Tuning protocol (fairness):** every method (baselines and agent) gets the same tuning budget (e.g., 20 configurations), tuned on **train**, selected on **validation**, reported on **test**. Save the chosen parameters and full sweeps.

---

## 9. RL specification (PPO from scratch)

**Starting point:** adapt the owner's existing PPO at `D:\Atharva\Implementations\From_Scratch\Reinforcement_Learning\PPO\train_ppo.py` (and notebooks there). It is single-environment; here it must be **vectorised over many function-days**.

**Formulation**
- One shared policy for all functions (parameter sharing), categorical over 6 actions.
- **Observation** (per function, per minute; all normalised with a running mean/std): demand summaries over windows 1/5/15/60 minutes (mean and max), trend (short minus long window mean), reference demand `R`, current pool `P[t-1]` and `P/R` ratio, minutes since last non-zero demand (log-scaled), sin/cos time-of-day, day-of-week (sin/cos), demand in the same window yesterday (with a "not available" flag on day 1), trigger type (one-hot), log memory, log average duration, function-scale `N[f]` (log).
- **Reward:** Section 7.
- **Networks:** policy MLP 2x64 tanh; separate value MLP 2x64. Orthogonal init; small final-layer gain on the policy head.
- **Algorithm:** PPO-clip with GAE. Starting points *(tune, do not trust)*: gamma 0.99, lambda 0.95, clip 0.2, Adam lr 3e-4, 4 epochs, 8 minibatches, entropy coef 0.01, grad-clip 0.5, 256 parallel function-days per iteration, rollout length 1440, about 200 iterations. Estimated cost: tens of minutes to about an hour on CPU *(estimate; validate)*.
- **Running observation normalisation** (learned from experience in a previous project: without it, large-scale features destabilise training). Save normaliser statistics inside the checkpoint.
- **Cost-conditioned option (stretch):** feed `log(rho)` as an input and sample `rho` per episode to get a whole Pareto front from one policy.
- **Sanity environment (required before real data):** a synthetic sinusoidal demand where the best headroom is known analytically. PPO must learn it (unit test for the RL loop).
- **Seeds:** at least 3, report mean and confidence interval.

**Ablations**
- no seasonal feature; no time-of-day; no pool in state
- `gamma = 0` (bandit) vs `gamma = 0.99`
- `kappa = 0` vs `kappa > 0`
- action set: relative-only (no absolute 0/1) vs full
- concurrency `c = 1` vs `c = 4`

---

## 10. Evaluation protocol and metrics

**Metrics (per function, then aggregated)**
- `cold_pct = 100 * sum(cold) / sum(n)` (approximation: each cold instance start delays at least one invocation; document it).
- `wasted_GBmin = sum(idle * mem/1024)`; also report normalised to fixed-10-minute on the same set (mirrors the paper).
- Paper-style summary: **75th percentile across functions of `cold_pct`**.
- `churn = sum(|P[t]-P[t-1]|)` and a flapping rate (direction reversals per hour).
- Mean pool utilisation.
- All with **bootstrap confidence intervals over functions** and **mean +/- std over agent seeds**.

**Comparisons to always produce**
- Pareto plot (Section 11, V4) with baseline sweeps as curves and agent points from the `rho` sweep.
- Per-class table (rare / predictable / bursty / hot).
- Held-out days and held-out functions reported separately.
- Distribution-shift checks: weekday vs weekend; train-class vs unseen-class.

**Honesty rules:** equal tuning budget; never pick hyperparameters on test; report failures; keep the exact command and seed for every reported number.

---

## 11. Visualisations (a first-class deliverable)

The owner wants to **see the agent learning the state and policy**, and to have shareable assets. Use matplotlib (+ Pillow for GIFs); keep a consistent style (agent highlighted, baselines muted, colour-blind-safe palette, readable on mobile). All scripts read logs/checkpoints and write to `reports/figures/`.

| ID | Figure | Purpose / what it shows | Inputs |
|---|---|---|---|
| **V1** | **Training dashboard** | reward, cold %, idle GB-min vs training iteration with baselines as horizontal reference lines; policy entropy; **stacked-area of the 6 action frequencies over training** (shows the agent "discovering" which actions matter) | training logs |
| **V2** | **Checkpoint replay ("watch it learn")** | the *same held-out function-day* replayed by the agent at checkpoints (untrained / 10% / 30% / final) as small multiples: demand line, agent pool line, baseline pool lines; shade cold-start minutes red and idle minutes blue. Also render as a **GIF that scrubs through checkpoints**. This is the hero asset. | checkpoints + one function-day |
| **V3** | **Policy heatmap** | expected headroom factor (or action probabilities) as a function of (recent trend x time-of-day), other features sampled from the real state distribution; one panel per function class. Shows *what rule the agent learned*. | final checkpoint + state samples |
| **V4** | **Pareto plot** | x = wasted memory (normalised to fixed-10), y = cold-start % (75th percentile). Baseline parameter sweeps as curves; agent `rho`-sweep as points with error bars over seeds; oracle as the floor. | evaluation tables |
| **V5** | **Per-class bars** | cold % and idle by class for the best baseline vs agent, with CIs. **Shows where RL wins and loses.** | evaluation tables |
| **V6** | **Live fleet view** | grid of the demo functions: demand vs running pods over time; colour = under-provisioned / ok / over-provisioned; scrolling event log of scaling actions (time, from -> to, reason, which safety rule fired) | live logs |
| **V7** | **Feature importance** | permutation importance of state features on return; "what the agent looks at" | final checkpoint + eval set |
| **V8** | **Live K8s demo panel** | pod count vs request rate over time from the live run, real pod-ready-latency histogram, agent vs baseline controllers side by side on the same replayed trace slice | live logs |
| **V9** | **Data EDA** | rate distribution (log-log), diurnal pattern, inter-arrival CV distribution, class counts; reproduces the paper's workload facts as a data sanity check | processed data |

**Public-facing set:** V2 GIF (hero), V4, V5, V8, plus one architecture diagram. Keep every figure reproducible with one command.

---

## 12. Live Kubernetes integration (kind)

**Goal:** show the trained policy actually scaling real Deployments. This is an **integration and learning demo, not the benchmark** (benchmark numbers come from the simulator).

**Cluster:** `kind` on Docker Desktop, 1 control-plane + 1 worker, config in `live/kind-config.yaml`. Install `kind` (free single binary; `winget install Kubernetes.kind` *(verify)* or download from its GitHub releases; **ask the owner before installing**). `kubectl` v1.36 is already installed.

**Workload**
- About **5 demo functions** chosen to span classes (one predictable, one bursty, one rare, one hot, one medium). Cap each at about 6 replicas *(memory estimate; validate in the spike)*.
- **Worker image** (`live/worker/`): tiny HTTP service; handler burns CPU or sleeps for the function's duration; single-threaded so concurrency = 1; `/healthz` for the readiness probe. To mimic a cold start, add an **emulated startup delay** env var (e.g., 2 s) and *state clearly that it is emulated*. Also measure the real container/pod startup time.
- **Load replayer** (`live/replayer.py`): reads a function's per-minute counts, compresses time by `S` (default 10, so 1 simulated minute = 6 s), scales volume so the peak fits the replica cap, issues requests with Poisson arrivals inside each simulated minute, and exposes a `/metrics` endpoint (requests in window, in-flight, latency percentiles). **Skip Prometheus** in v1 (optional stretch).
- **Time-compression caveat:** pod start takes seconds and does not compress. At `S = 10` a start should be about half a simulated minute or less; verify in the spike and adjust `S`.

**Controller** (`live/controller.py`): loop every simulated minute -> `K8sBackend.observe()` -> `FeatureBuilder` -> `Autoscaler.decide()` -> `SafetyWrapper` -> `K8sBackend.apply()` which **patches the Deployment `scale` subresource** via the `kubernetes` client. Modes: `--dry-run`, `--policy {agent,fixed,hpa}`. Log every decision as JSON lines (time, function, observed metrics, raw decision, safe decision, reason).

**Live experiment:** replay the *same* 2-hour trace slice through (a) fixed keep-alive controller, (b) HPA-style controller, (c) the RL agent; record cold/idle equivalents, p95 request latency, pod-ready latency, churn. Compare visually (V6, V8). **Stretch:** also compare against the **native Kubernetes HPA** (`autoscaling/v2`, needs `metrics-server` on kind with `--kubelet-insecure-tls`).

**Kubernetes concepts the owner will learn here:** Pod, Deployment, ReplicaSet, Service, readiness probes, the `scale` subresource, kubeconfig/contexts, `kubectl get/describe/logs/scale`, kind clusters, and (stretch) RBAC with a least-privilege ServiceAccount when the controller is containerised.

---

## 13. Safety wrapper (real-autoscaler practice)

`SafetyWrapper` wraps any `Autoscaler` and enforces, in order:
1. **Bounds:** `min_instances <= P <= max_instances` (per function, from config).
2. **Rate limit:** cap scale-up per step (e.g., at most 2x or +N) and scale-down (e.g., at most -25% per step).
3. **Cooldown / stabilisation:** do not scale down within X minutes of a scale-up.
4. **Out-of-distribution guard:** running z-scores of the observation vs training statistics; if more than k features exceed a threshold for m consecutive steps, **fall back to the HPA-style rule** and log the reason.
5. **Kill switch:** env var / flag that forces the fallback policy.
Every clamp or fallback is recorded with a reason string. Property-based tests (Section 14) check the invariants.

---

## 14. Testing strategy

- **Unit tests** per module, run with `pytest`; every brick lists its "Done when" tests.
- **Hand-computed fixtures:** tiny traces where cold/idle/reward are computed by hand.
- **No-lookahead property test:** perturbing demand at minute `t` or later must not change any feature or decision at minute `t`.
- **Parity tests:** single-function env vs vectorised env; sim `FeatureBuilder` vs live `FeatureBuilder` on identical history.
- **RL-loop sanity:** PPO solves the synthetic sinusoid; return improves and matches the analytic optimum within tolerance.
- **Determinism:** same seed gives the same metrics.
- **Safety invariants:** property tests (never exceeds max, never violates rate limit, fallback triggers on OOD input).
- **Leakage tests:** no function or day overlap across splits.
- Live components get **smoke tests** (dry-run controller, `kubectl` calls mocked), plus a manual checklist for the real cluster.

---

## 15. Brick-by-brick plan

Format per brick: **Goal / Build / Logic / Done when / Pitfalls**. Tick the box when done.

**How each brick is delivered (LLM to owner).** Follow these steps every time:
1. Restate the brick's Goal and Done-when in two or three lines.
2. Explain the logic and the key design decision(s) in plain language: *why* this design, and what the alternatives were.
3. Give the **tests first** as a code block, with the file path to save it at and the exact command to run. Expected result: the tests fail.
4. Give the **implementation in small annotated blocks**, each with its target file path and a short explanation. The owner types these in; do not touch the project folder yourself.
5. Say exactly what to run next and what output means success. List the likely errors and their fixes.
6. Add one or two "check your understanding" prompts or small experiments (for example: change a parameter and predict what happens).
7. **Stop and wait for the owner's result** (test output or his own version of the code) before starting the next brick. If something fails, debug from his actual output.
8. Before sending, verify the code in a scratch folder outside the project, and report what was verified.

### Phase 0: Foundation and data

- [ ] **B0 Scaffold and environment**
  - *Goal:* reproducible project skeleton.
  - *Build:* folders (Section 5), `requirements.txt` (numpy, pandas, pyarrow, matplotlib, pillow, torch (CPU), pytest, kubernetes, pyyaml), a seed helper, a config loader, `.gitignore` (data/, reports/, checkpoints).
  - *Logic:* fix seeds and configs from day 1 so every number is reproducible.
  - *Done when:* `pytest` runs (with one trivial test); `python -c "import torch, pandas"` works in a fresh venv.
  - *Pitfalls:* Windows paths (use `pathlib`); do not commit raw data.

- [ ] **B1 Acquire the data**
  - *Goal:* raw Azure 2019 files on disk.
  - *Build:* a small download script; check the archive size first and confirm with the owner.
  - *Logic:* CC-BY data; record source URL, date, file count and hash in a `DATA_CARD.md`.
  - *Done when:* 14 invocation files, 14 duration files, 12 memory files present; row counts logged.
  - *Pitfalls:* large archive; `.tar.xz` extraction on Windows.

- [ ] **B2 Loader and EDA**
  - *Goal:* clean per-function arrays plus metadata.
  - *Build:* parse invocation CSVs into `(function, day, minute)` arrays (`uint16/float32`), join durations and memory, compute per-function stats and **class labels** (Section 6), save processed parquet/npz for a **stratified subset** (e.g., about 3-5k functions).
  - *Logic:* the full trace is large; sampling stratified by class keeps rare and hot functions represented.
  - *Done when:* V9 figures produced; paper facts (rare-function share, invocation concentration, predictable share) reproduced approximately; class thresholds frozen in config.
  - *Pitfalls:* missing memory days; app-level memory joined to function-level rows; huge dense arrays.

- [ ] **B3 Demand builder**
  - *Goal:* `D[f,t]` from Section 7.
  - *Build:* `demand.py` with `c` as a parameter.
  - *Done when:* hand-computed examples pass (n=0 -> 0; tiny n with long duration -> at least 1; rounding correct); `c=4` reduces demand as expected.
  - *Pitfalls:* averaging duration per day vs global; zero-duration rows.

- [ ] **B4 Splits and leakage guard**
  - *Goal:* frozen, seeded train/val/test splits by day and by function.
  - *Build:* `splits.py`, saved ids.
  - *Done when:* leakage tests pass; class proportions similar across splits; splits saved to disk and never regenerated.

### Phase 1: Simulator and metrics

- [ ] **B5 Single-function simulator**
  - *Goal:* exact reference implementation of Section 7.
  - *Build:* `sim/env.py` (`reset`, `step(action)`), reward and metrics per step.
  - *Done when:* hand-made traces give exact `cold`, `idle`, reward; action mapping (0/1/0.5R/1R/1.5R/2R) correct; `P` clamped.
  - *Pitfalls:* off-by-one in "information up to t-1"; ceil of zero.

- [ ] **B6 Vectorised environment**
  - *Goal:* N function-days in parallel, fast.
  - *Build:* `sim/vec_env.py` with a Gym-*like* API written by hand; pre-computed feature tensors indexed per step.
  - *Done when:* parity test vs B5 on random traces (identical rewards/metrics); throughput measured and logged *(estimate: millions of env-steps/minute)*.
  - *Pitfalls:* batching functions of different lengths; padding; determinism.

- [ ] **B7 FeatureBuilder (shared by sim and live)**
  - *Goal:* the single source of truth for the state.
  - *Build:* `features/builder.py` operating on a history buffer; running normaliser.
  - *Done when:* **no-lookahead property test** passes; feature vector shape and ordering documented; the same builder works on both a trace-backed history and a live rolling buffer (test with synthetic history).
  - *Pitfalls:* silent lookahead via array slicing; day-1 seasonal feature masking.

- [ ] **B8 Metrics and evaluation harness**
  - *Goal:* one function that evaluates any policy on any split and returns all metrics with CIs.
  - *Build:* `sim/metrics.py`, bootstrap CI, Pareto helpers, per-class aggregation, saved tables.
  - *Done when:* oracle gives zero cold/idle; "never scale" and "never unload" produce the expected extremes; CIs computed.

### Phase 2: Baselines (build them BEFORE any RL)

- [ ] **B9 Fixed keep-alive, never-unload, oracle** - *Done when:* K sweeps behave monotonically (larger K -> fewer cold, more idle); handcrafted trace checks pass.
- [ ] **B10 Hybrid histogram** - read paper section 5; implement histogram, representativeness test, pre-warm/keep-alive, fixed fallback. *Done when:* unit tests on synthetic periodic and irregular arrivals; deviations from the paper documented.
- [ ] **B11 HPA-style reactive** - formula, tolerance, stabilisation window, min/max. *Done when:* known-input tests match hand calculation (including tolerance dead-band and scale-down stabilisation).
- [ ] **B12 Seasonal predictive** - yesterday-window forecast + margin + reactive fallback on day 1. *Done when:* perfectly periodic synthetic demand gives near-zero cold with small idle.
- [ ] **B13 Tuning and the baseline report**
  - *Goal:* fair, equal-budget baseline results.
  - *Build:* `baselines/tune.py` (grid/random search, same budget for all), save sweeps and chosen params.
  - *Done when:* baseline Pareto curves and per-class table exist. **Gate S1:** hybrid < fixed-10 on cold % at comparable memory (directional match to the paper). If not, debug the simulator now.

### Phase 3: RL agent

- [ ] **B14 Vectorised PPO and the sanity environment**
  - *Goal:* a correct PPO on the vec env.
  - *Build:* `rl/ppo.py`, `networks.py`, `normalize.py`; adapt the owner's PPO; a synthetic sinusoid environment with a known optimal headroom.
  - *Done when:* PPO reaches near-optimal return on the sinusoid; loss/entropy/KL logged; a checkpoint save/load round-trips exactly.
  - *Pitfalls:* GAE across time per env with truncated horizons; advantage normalisation per batch; checkpoint must include normaliser stats.

- [ ] **B15 Train on real traces**
  - *Goal:* trained agents for `rho in {3, 10, 30}`, 3 seeds each.
  - *Build:* `rl/train.py` with the same tuning budget as baselines; logging for V1; periodic checkpoints (untrained, 10%, 30%, 100%) for V2.
  - *Done when:* validation return improves over training; logs and checkpoints saved; the equal-budget record is written down.

- [ ] **B16 Evaluation and ablations**
  - *Goal:* the honest comparison.
  - *Build:* `rl/evaluate.py`; run Section 9 ablations and Section 10 comparisons on held-out days and functions.
  - *Done when:* tables for overall and per-class results with CIs; ablation table; a written 5-line "where RL wins / loses / why" note.

- [ ] **B17 Learning-behaviour visualisations**
  - *Goal:* V1-V5, V7, V9 with one command.
  - *Done when:* all figures regenerate from logs/checkpoints; V2 GIF renders; a stranger can read the Pareto plot without explanation.

### Phase 4: Integration on Kubernetes

- [ ] **B18 Autoscaler interface and SafetyWrapper**
  - *Goal:* the deployable seam.
  - *Build:* `scaler/interface.py`, `safety.py`, `policy_adapter.py` (loads a checkpoint, uses `FeatureBuilder`).
  - *Done when:* property tests for bounds, rate limit, cooldown, OOD fallback and kill switch pass; the adapter reproduces the sim policy's decisions exactly on a replayed history.

- [ ] **B19 kind cluster, worker service, replayer (spike first)**
  - *Goal:* a working local cluster with real scalable workloads.
  - *Build:* install `kind` (ask first), `kind-config.yaml`, worker image, one Deployment, replayer; manually `kubectl scale` and measure real pod-ready time.
  - *Logic:* the spike decides replay speed `S`, replica caps, and how many functions fit in memory.
  - *Done when:* one function scales 1 -> 5 -> 1 under replayed load; pod-ready latency measured and recorded; memory footprint recorded.
  - *Pitfalls:* Docker Desktop memory limit; image loading into kind (`kind load docker-image`); readiness probe timing.

- [ ] **B20 Controller loop (K8sBackend)**
  - *Goal:* metrics -> features -> policy -> safety -> `scale` patch.
  - *Done when:* `--dry-run` logs correct decisions; a live run with the fixed-keep-alive policy scales a Deployment as expected; JSON decision log written; failures (API errors, missing metrics) handled without crashing.

- [ ] **B21 Live experiment and dashboards**
  - *Goal:* the same 2-hour trace slice replayed under fixed / HPA-style / agent controllers.
  - *Done when:* V6 and V8 generated from real logs; a comparison table (p95 latency, pod-ready latency, churn) exists; native HPA comparison attempted (stretch) or explicitly deferred.

### Phase 5: Packaging

- [ ] **B22 README, results, and post**
  - *Goal:* everything a reader needs in five minutes.
  - *Build:* README (problem, architecture diagram, results, limitations, how to reproduce, data citation), hero GIF, short demo video, LinkedIn/X thread outline (lead with the problem and the numbers; include one honest limitation).
  - *Done when:* a fresh clone reproduces the main figure with documented commands.

---

## 16. Timeline and scope-cut order

- **Day 1:** B0-B8 (data, simulator, features, metrics).
- **Day 2:** B9-B13 (baselines and the baseline report), start B14.
- **Day 3:** B14-B17 (PPO, training, evaluation, visualisations).
- **Day 4:** B18-B21 (interface, safety, kind, controller, live experiment).
- **Day 5:** B22 plus buffer.

Three days is tight; five evenings is realistic. **If behind, cut in this order:** native-HPA comparison -> cost-conditioned policy -> feature-importance (V7) -> containerised controller/RBAC -> some ablations -> fewer live functions. **Never cut:** baselines with equal tuning, held-out evaluation, V2/V4/V5, the safety wrapper.

---

## 17. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Agent does not beat a tuned baseline (predictable functions are easy) | Report per class; the pipeline, visuals and honest analysis still have value; test `gamma=0` ablation |
| Simulator fidelity (minute bins, average durations, concurrency assumption) | State limitations up front; sensitivity to `c`; live demo as a sanity check |
| Reward-weight sensitivity | Pareto sweep over `rho`; never a single number |
| Overfitting to specific functions | Split by function and by day; per-class reporting |
| Data too large / slow to load | Stratified subset; store compact arrays |
| Docker/kind resource limits on a laptop | Spike in B19; fewer functions and lower replica caps |
| Time compression vs real pod start | Choose `S` from measured pod-ready time; label the emulated delay |
| Kubernetes learning curve | Do B19 as a small spike before building the controller; learn commands manually first |
| Free-tier limits (if Colab/Kaggle ever used) | Not required for this design |
| Benchmark-paper warning that RL often loses to calibrated baselines on predictable loads | Equal tuning budgets; per-class analysis; honesty rules (Section 10) |

---

## 18. Open decisions to confirm with the owner (defaults in bold)

1. Action set: **6 actions (0, 1, 0.5R, 1R, 1.5R, 2R)**, where "1" was added so rare functions can keep one warm. Confirm.
2. Concurrency per instance `c`: **1**, sensitivity at 4.
3. Reference demand `R`: **max of last 5 minutes**.
4. Start/churn cost `kappa`: **0.1**, ablate with 0.
5. Function subset size: **3-5k stratified functions**.
6. `rho` values: **{3, 10, 30}**.
7. Live demo: **5 functions, replay speed 10x, 2-hour slice**; native-HPA comparison as **stretch**.
8. Who writes the code: **the LLM provides complete, annotated, tested code in chat one brick at a time (tests first); the owner types it in and runs it himself. The LLM does not edit the project folder unless he asks.** (Decided 2026-09-26.)
9. Project home: `D:\Atharva\Implementations\From_Scratch\RL_Autoscaler` (this folder).

---

## 19. Environment facts and existing assets

- **OS/tools:** Windows 11 (PowerShell and Git Bash available). Docker Desktop **running** (about 12 CPUs, about 7.9 GB allocated to Docker). `kubectl` v1.36.1 installed. **`kind` not installed** (to be installed with the owner's approval). WSL present. No local GPU.
- **Hardware:** Ryzen 5 5500U, 15.3 GB RAM.
- **Existing RL code to adapt:** `D:\Atharva\Implementations\From_Scratch\Reinforcement_Learning\` -> `PPO/` (`train_ppo.py`, CartPole/LunarLander/VizDoom notebooks), `CartPole_A3C/`, `DQN-Taxi/`, `Flappy_Bird_DQN/`, `SAC/`.
- **Parallel project (context only):** `Fly_Brain` (a frozen fly-connectome network trained with REINFORCE on Flappy Bird). Lessons carried over: test the RL loop on a toy problem first; build strong baselines before the fancy model; keep honest controls; use running observation normalisation; write tests and keep experiments reproducible.

---

## 20. References

- Shahrad et al., **Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider**, USENIX ATC 2020. https://www.usenix.org/system/files/atc20-shahrad.pdf (also arXiv 2003.03423). *Cite when using the data or the hybrid-histogram baseline.*
- Azure Public Dataset (Functions 2019): https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md (CC-BY).
- Azure Functions 2021 invocation trace (optional, per-invocation records, modified timestamps): https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md
- Rzadca et al., **Autopilot: workload autoscaling at Google**, EuroSys 2020 (vertical autoscaling context). https://dl.acm.org/doi/10.1145/3342195.3387524
- **When Does Deep RL Beat Calibrated Baselines? A Benchmark Study on Adaptive Resource Control** (evaluation methodology: calibrate baselines, equal budgets, diverse workloads). https://arxiv.org/pdf/2605.26418 *(summary read; full text not verified)*
- RL-based Kubernetes autoscaling literature (Gym-HPA, AWARE, Q-learning/Dyna-Q, forecast-based approaches); a starting survey link: https://link.springer.com/chapter/10.1007/978-3-032-10344-4_3 *(only snippets read; treat reported gains as unverified and non-transferable across workloads)*
- Cold-start work referenced in searches: Taming Cold Starts with MPC https://arxiv.org/html/2508.07640v1 ; RL cold-start reduction https://arxiv.org/pdf/2308.07541 *(not read)*
- Kubernetes HPA algorithm documentation (read the "algorithm details" section before B11).
