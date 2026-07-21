# Decision: shared experiment-tracking backend (MLflow/W&B) in `dscraft.core`?

- **Status:** Decided — defer. `tune` has a landed, real, iterative
  training loop (the one subpackage close to qualifying); no other
  subpackage does; the "at least two subpackages" bar for promoting a
  shared `dscraft.core` capability isn't met yet with only one.
- **Date:** 2026-07-21
- **Issue:** [gr3enarr0w/dscraft#35](https://github.com/gr3enarr0w/dscraft/issues/35)

## Summary

Issue #35 asks whether `dscraft.core` should expose a pluggable
experiment-tracking integration point (MLflow and/or W&B) for
training-loop-having subpackages. **Recommendation: defer, do not build
yet — but the reasoning is narrower than "no subpackage qualifies."**
The issue names three subpackages to check (`forecast`, `automl`, `eda`);
none of them has a real, iterative, multi-run training loop that produces
metrics worth tracking and comparing over time — the closest of the three
(`forecast.backtest()`) is a single evaluation call returning a plain
report object, not a training loop. But the issue's own named list is not
the complete set of ten subpackages, and this evaluation additionally read
`dscraft.tune` in full (`adapter.py`, `export.py`, `__init__.py`) even
though issue #35 doesn't name it — see part (a). **`dscraft.tune` does
have a landed, real, iterative training loop**:
`ProgrammaticAdapter.train_step()` runs a genuine forward/backward/
optimizer-step LoRA update per call and returns a `TrainStepResult(loss,
step)`; `examples/tune/lora_finetune_example.py` calls it in a 25-iteration
loop, producing an actual loss curve — precisely the multi-step metric
history shape experiment tracking exists for. So the accurate finding is:
**one** subpackage (`tune`) already has code shaped for tracking, not
zero. CLAUDE.md's "core stays thin, build shared infra only once two real
modules need it" rule requires landed consumers in *two* subpackages before
promoting a shared `dscraft.core` capability — one qualifying subpackage
isn't enough to clear that bar, so the recommendation is still to defer,
but explicitly because the two-subpackage gate isn't met (not because zero
subpackages have a real training loop). Separately, and just as
importantly: OpenTelemetry (already `dscraft.core`'s locked telemetry
schema) covers a *different* problem than experiment tracking, so this
would be a genuinely new, complementary capability if built — not a
duplicate of `dscraft.core.telemetry` — but that only matters once the
two-subpackage bar is actually met.

## (a) Which subpackages currently produce metrics/results, and how do they surface them today?

Checked every subpackage the issue names, by reading the actual code:

- **`forecast`** (`packages/dscraft/src/dscraft/forecast/backtest.py`):
  `backtest()` returns a `BacktestReport` dataclass — `metrics: list[SeriesMetric]`
  (one `SeriesMetric` per `(series, model)` pair: `mae`, `rmse`, `n_points`,
  `expected_points`), plus `mean_mae()`/`mean_rmse()` helpers and a
  `to_frame()` that renders the report as a plain pandas `DataFrame`. This
  is a **single evaluation call**, not a training loop — there is no
  concept of "epoch," "step," or a run that improves over iterations to
  track. It's the one closest candidate to something "trackable" among the
  subpackages checked, but its natural comparison unit (one backtest run
  per model/config, compared against another backtest run) is already
  served by comparing two `BacktestReport.to_frame()` outputs directly —
  there's no accumulation-over-time need visible in the actual code today.
- **`automl`**: `compile.py` exists (the `.compile()` → ONNX export path per
  the architecture doc), but there is no training-loop or metrics-reporting
  code in the subpackage today beyond what `compile.py` needs for its ONNX
  export. The issue's "automl's eventual training runs" is explicitly
  forward-looking — "eventual" is the issue's own word — not a description
  of code that exists.
- **`eda`**: `engine.py`/`sketches.py`/`associations.py`/`report.py`
  produce a one-shot **profiling report** (schema, null counts, HLL/KLL
  sketches, association matrix, rendered as a dependency-free HTML/Canvas
  report). This is not a training run in any sense — there's no metric that
  improves over iterations, no hyperparameter to sweep, no "run" concept at
  all. It's a snapshot of a static dataset. Experiment-tracking backends
  (run/step/metric-curve/hyperparameter-versioning tools) don't map onto
  this shape at all; this is arguably the weakest of the three cited
  candidates for needing tracking of any kind.
- **`tune`** (`packages/dscraft/src/dscraft/tune/adapter.py`,
  `export.py`, `__init__.py` — read in full for this evaluation, even
  though issue #35 doesn't name this subpackage): **this is the one
  subpackage that does have a real, iterative, multi-step training loop
  today.** `ProgrammaticAdapter` (the `BaseTrainingAdapter` "Adapter-Factory"
  implementation for in-process LoRA fine-tuning via `peft`+`transformers`)
  exposes `train_step(batch) -> TrainStepResult`, where each call runs a
  genuine forward pass, `loss.backward()`, and `optimizer.step()` against
  the LoRA-wrapped model, returning `TrainStepResult(loss: float, step:
  int)` with a monotonically-incrementing step counter. This is not a mock
  or a stub — `export.py`'s GGUF/MLX conversion functions are explicit
  `NotImplementedError` stubs, but `adapter.py`'s training path is real,
  working code. `examples/tune/lora_finetune_example.py` calls
  `train_step()` in a 25-iteration loop over a small synthetic corpus and
  prints the loss trajectory (`compute_loss()` before, `TrainStepResult.loss`
  at each step, `compute_loss()` after) — exactly the "loss curve across
  training steps" shape experiment tracking exists to log and compare
  across runs. What `tune` does *not* yet have: any notion of a "run"
  (multiple `prepare()`+training-loop invocations compared against each
  other), hyperparameter sweeps, or any tracking-backend call sites — it's
  a real training loop with no tracking integration today, not a
  training loop that already needs one to be built for a demonstrated pain
  point. That distinction matters for the revisit criteria in part (d): the
  loop exists and produces trackable metrics, but there is exactly one such
  subpackage, not two.

**Conclusion for (a):** every subpackage named in the issue (`forecast`,
`automl`, `eda`) surfaces its results today as a plain dataclass and/or
pandas DataFrame, with zero tracking-backend integration and no actual
multi-run, metric-over-iterations training loop — they produce one-shot
reports. This part of the issue's premise (about the three subpackages it
names) is correct. But `tune`, which the issue does not name, was checked
in this pass and **does** have a landed, real, iterative training loop
producing exactly the per-step metric history experiment tracking is for.
The accurate summary is therefore "one subpackage (`tune`) has code shaped
for tracking; the three subpackages the issue actually asked about do
not" — not a blanket "no subpackage qualifies."

## (b) MLflow vs. W&B: local-only compatibility

Per CLAUDE.md's multi-backend principle, *if* tracking is added, it must
support both as selectable backends, never one hard-coded choice. Both are
Tier-1 permissive per the issue (MLflow Apache-2.0, W&B client MIT) — no
LazyIsolate gating needed for the client libraries.

- **MLflow — local-only: qualifies trivially, but two separate mechanisms
  must not be conflated: client-side logging vs. the viewer process.**
  These are genuinely two different things:
  - **Logging metrics (the client side) never starts a server.** A Python
    process calling `mlflow.log_metric(...)` (with no tracking URI
    configured) writes tracking data directly to a local `./mlruns`
    directory on disk — no server process, no account, and no network
    dependency at any point. This has been MLflow's default, zero-config
    client behavior since its earliest releases and remains so. The
    idiomatic way to do this is to wrap the call in an explicit run context,
    per MLflow's own docs:
    ```python
    import mlflow

    with mlflow.start_run():
        mlflow.log_param("lr", 1e-2)
        mlflow.log_metric("loss", loss, step=step)
    ```
    Note for precision: MLflow's fluent API does not *require* an active
    run to already exist — per MLflow's own API reference, `log_metric()`
    "will create a new active run" if none is active when called. But that
    auto-created run is not automatically ended, so relying on this
    implicitly (rather than wrapping calls in an explicit
    `mlflow.start_run()` block, as MLflow's own tracking-quickstart docs
    recommend) leaves run lifecycle management up to the caller. If this
    is ever implemented, the `dscraft` integration should always use an
    explicit `mlflow.start_run()` context, not rely on the implicit
    auto-created run.
  - **Viewing runs (the `mlflow ui` / `mlflow server` side) is a separate,
    optional local server process that does need to be started.** `mlflow
    ui` (or `mlflow server`) is its own command that must be explicitly run
    to browse logged runs in a dashboard — it is not something that
    logging metrics starts implicitly, and it is not "no server" the way
    client-side logging is. As of MLflow 3.7, this server's default backend
    store changed from the local `./mlruns` file store to a local SQLite
    database (`sqlite:///mlflow.db`) unless `--backend-store-uri ./mlruns`
    (or an equivalent `MLFLOW_TRACKING_URI`) is set to opt back into the
    legacy file-store behavior; either way, it remains a local-only HTTP
    server process with no external network dependency and no account
    requirement, which is still a clean fit for CLAUDE.md's local-only
    constraint — it just isn't correct to describe it as "no server process
    ... required at any point," since a server process is exactly what
    `mlflow ui`/`mlflow server` is, even though it needs no external
    network access.
- **W&B — local-only: qualifies, with one documented caveat that must be
  called out rather than assumed away.** `wandb.init(mode="offline")` is
  documented by W&B itself (their official support docs, "difference
  wandbinit modes") to write run data to a local file only:
  *"I don't want to depend on the network to send results to your servers
  while executing local operations" ... "wandb.log ... does not block
  network calls."* That's the intended, documented behavior, and it means
  no metric/artifact data leaves the machine unless a separate, explicit
  `wandb sync` command is run later by the user. **However**, research for
  this evaluation surfaced a real, filed W&B GitHub issue
  (`wandb/wandb#2701`, "With WANDB_MODE=offline python client still try to
  sync results") reporting that even with offline mode set, the client
  attempted outbound HTTPS connection retries to W&B's own API endpoint
  (visible in `urllib3` connection-retry warnings) before failing/timing
  out — i.e., no data was exfiltrated (the connection attempts failed), but
  the client did *attempt* outbound network activity in offline mode on at
  least one reported version, which contradicts a strict "fully quiet on
  the wire" expectation for an air-gapped environment. W&B also runs a
  local background service process (`wandb-service`) on `wandb.init()`
  regardless of mode, which listens for local filesystem changes — not
  itself a network concern, but additional local process/resource
  overhead MLflow's plain file-append approach doesn't have.
  **Plain finding: W&B's offline mode does not intentionally phone home —
  by design and by W&B's own documentation, `wandb.log()` never blocks on
  or requires network access — but it is not proven quiet-by-construction
  the way MLflow's local file store is; there is at least one credible,
  filed report of connection *attempts* (not data leaks) still occurring
  in offline mode.** If W&B is ever adopted as a selectable backend here,
  this should be re-verified against the exact pinned `wandb` version at
  implementation time (and ideally tested inside the same sandboxed/
  network-denied environment `dscraft.core.sandbox` already provides for
  `security`/`agent`) rather than trusted on documentation alone.

## (c) Does OpenTelemetry already cover this need? (the central question)

**No — OTel and experiment tracking solve different problems, and a
tracking integration would be complementary, not a competing duplicate
system, PROVIDED it is built to reuse the existing OTel schema rather than
invent a second, parallel metric-naming/schema convention.** This
distinction is worth stating precisely because getting it wrong either way
has a real cost: treating them as redundant would wrongly block a
legitimate future capability; treating them as unrelated and building a
second schema from scratch would violate the architecture doc's own
"shared architecture" decisions (CLAUDE.md: OTel GenAI semantic
conventions are *the* shared schema across security reports, agent
trajectories, and ML leaderboards).

Concretely, reading `dscraft/core/telemetry.py` (311 lines) end to end:

- OTel here is **span/trace-oriented, in-process, and export-optional**.
  `get_tracer()`/`genai_span()`/`set_ml_metric()` create OTel spans and
  attach attributes (`ml.metric.<name>`) to them; without an application-
  configured SDK/exporter, the tracer is a documented no-op — "spans are
  created and attributes/events are accepted, but nothing is exported
  anywhere." Its entire design center is *observability of a running
  process* (a single security probe run, a single agent trajectory, one
  in-flight ML operation), correlated via span parent/child relationships
  within that one execution.
- OTel has **no run/experiment grouping model, no hyperparameter-logging
  concept, no artifact/model-versioning concept, and no cross-run
  comparison UI** — none of which is a gap or oversight in
  `telemetry.py`; it's simply outside OTel's problem domain. `set_ml_metric`
  attaches exactly one metric value to one span at one point in time; there
  is nothing in this module (or in OTel generally) that represents "this
  metric's value across 200 training steps of run A, and how that compares
  to run B and run C, alongside a versioned copy of the model artifact each
  run produced." That is precisely what MLflow/W&B exist to do, and it's
  materially different work: run/experiment identity, a metric-history
  time series *per named run*, hyperparameter key-value logging tied to
  that run, and a UI/API for browsing and comparing many runs.
- Because of that gap, if a real training-loop need shows up later
  (see (d)), the correct design is **not** a second, independent
  metric-naming scheme bolted directly onto MLflow/W&B calls scattered
  through subpackage code — that would indeed be the "second parallel
  telemetry system" this issue rightly worries about. The correct design
  is to keep `dscraft.core.telemetry`'s existing `ml.metric.*` /
  `genai_span` schema as the **single source of truth for what a metric is
  called and what a span represents**, and let an experiment-tracking
  integration be a *consumer* of that same schema — e.g. an OTel
  `SpanProcessor`/exporter that, when installed and configured by the
  calling application, translates `ml.metric.*` span attributes and
  `genai_span` boundaries into `mlflow.log_metric(...)`/`wandb.log(...)`
  calls under the hood. This is the standard, well-established OTel
  pattern (a span processor is exactly OTel's designed extension point for
  "do something else with these spans/attributes") and it means
  `dscraft.core.telemetry`'s existing attribute names and span-naming
  conventions never get a second, competing definition — MLflow/W&B become
  one more *sink* for the same events, not a second event vocabulary.

**Answer to (c), stated plainly: OTel does not already cover the
experiment-tracking need (they solve different problems), but an
integration should be built as a translator/exporter on top of the
existing OTel schema, not as an independent parallel system with its own
metric names and its own instrumentation call sites sprinkled through
subpackage code.**

## (d) Recommendation: defer, with explicit revisit criteria

**Do not build an experiment-tracking integration now**, in `dscraft.core`
or anywhere else. Two independent reasons converge on the same answer —
note reason 1 is now more precise than a blanket "zero real modules,"
because part (a)'s `tune` investigation changes the count from zero to one:

1. **CLAUDE.md's "two real modules" bar isn't met at one.** Per part (a),
   `tune` already has a real, iterative, multi-step training loop
   (`ProgrammaticAdapter.train_step()`, exercised across 25 steps in
   `examples/tune/lora_finetune_example.py`) that produces exactly the
   per-step loss history experiment tracking exists to capture. The three
   subpackages the issue actually names (`forecast`, `automl`, `eda`) do
   not — they produce one-shot reports, not metric-over-iterations training
   history. So there is now **one** concrete, landed consumer with a
   training loop shaped for tracking, not zero — but CLAUDE.md's bar is
   *two* real modules before promoting shared `dscraft.core` infrastructure,
   and `tune` alone doesn't clear it. Building this now, off a single real
   consumer, would mean designing an integration point against only one
   concrete shape (and one hypothetical second shape), which is exactly the
   premature-abstraction pattern CLAUDE.md's shared-infrastructure rule
   (and this same reasoning already applied in the #14 evaluation above)
   exists to prevent.
2. **The right integration shape depends on a *second* real training loop
   existing, not just the first.** `tune`'s loop tells us one concrete
   shape (`TrainStepResult(loss, step)` per LoRA training step, no run/sweep
   concept yet), but a shared interface designed off one data point risks
   overfitting to `tune`'s specific shape. Whether the eventual integration
   point should be "wrap `genai_span`/`set_ml_metric` with an optional
   exporter" (part (c)'s recommended shape) or something else entirely can
   only be validated with confidence once a *second* subpackage (e.g. a
   future `automl` hyperparameter-search implementation) has its own real,
   independently-arrived-at metric-history shape to compare `tune`'s
   against. Designing the exporter/translator shape today, off `tune` alone,
   risks building an interface that quietly bakes in `tune`-specific
   assumptions instead of a genuinely general one.

### Revisit criteria (concrete, so this isn't re-litigated from scratch)

Revisit this decision when **all** of the following are true:

- **At least two distinct `dscraft` subpackages** — not one — have landed
  real, iterative training-loop code that each produces a metric history a
  user would plausibly want to compare across multiple runs. `tune`'s
  `ProgrammaticAdapter.train_step()` loop already satisfies this for one
  subpackage as of this evaluation; a second landed candidate is still
  needed (most plausibly a future `automl` hyperparameter-search
  implementation — not a one-shot report like today's
  `forecast.backtest()` or `eda` profiling, and not a second, unrelated
  training loop bolted onto `tune` itself, which would still only count as
  one subpackage). One qualifying subpackage, however real its loop is, is
  explicitly **not sufficient** to promote a shared `dscraft.core.tracking`
  capability — this is the gate CodeRabbit's review flagged as needing to
  be explicit, and it is restated here precisely so a future single-
  subpackage justification doesn't slip through.
- At least one of those two (or a third) subpackage has an actual,
  expressed need to *compare* runs over time (not just log a single run's
  final metrics) — e.g. comparing LoRA fine-tuning runs across
  hyperparameter sweeps, or AutoML model-selection runs across candidate
  configs.
- The OTel-schema-layering requirement from part (c) still holds at that
  time: any eventual `dscraft.core.tracking` integration must be built as a
  consumer/exporter on top of the existing `dscraft.core.telemetry`
  `ml.metric.*`/`genai_span` schema, not a second, independent
  metric-naming convention. This requirement does not loosen just because
  the two-subpackage bar above is met.

At that point, don't treat this document as having pre-decided the
outcome — the specific tracking approach must itself be evaluated fresh
once the gate above is actually met, not assumed from today's guesswork.
That evaluation should determine, at minimum: which backend(s) to
support (per CLAUDE.md's multi-backend principle, this evaluation should
not rule out supporting more than one of MLflow/W&B as selectable
options, but whether it's one, both, or something else entirely is an
open question for that future evaluation, not a decision made here);
what interface shape best fits whatever the second real training-loop
subpackage's metric-history shape turns out to be (part (c)'s
OTel-span-processor/exporter framing is one plausible direction worth
weighing then, not a committed design); and whether this capability
belongs in `dscraft.core` at all, versus living elsewhere, versus not
being built as shared infrastructure at all. Two things from this
evaluation should carry forward unconditionally, because they're
already-locked constraints rather than tracking-specific choices: any
implementation must ship behind its own opt-in extra (never a base
dependency of `core` or any subpackage extra, per the issue's own
constraint), and it must not invent a second, parallel metric-naming
convention alongside `dscraft.core.telemetry`'s existing OTel schema
(per part (c) and CLAUDE.md's locked "OTel is the shared telemetry
schema" decision).

## What was NOT done in this pass

This issue is evaluation-only per its own text and per this task's
instructions. No tracking integration code was written; `dscraft.core`,
`forecast`, `automl`, `eda`, `tune`, and `pyproject.toml` are unchanged by
this evaluation. `dscraft.tune`'s source (`adapter.py`, `export.py`,
`__init__.py`) was read in full purely to answer part (a) accurately — no
tracking hooks, `mlflow`/`wandb` calls, or any other code were added to
`tune`, even though its training loop turned out to be real and
tracking-shaped; implementing an integration against it is explicitly out
of scope for this evaluation-only pass and is gated on the revisit
criteria in part (d).
