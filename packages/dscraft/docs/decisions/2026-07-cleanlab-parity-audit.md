# 2026-07 cleanlab parity audit for `dscraft.clean`

- **Status:** Accepted (decision + one small implementation landed; two gaps deferred to follow-up issues)
- **Issue:** [gr3enarr0w/dscraft#18](https://github.com/gr3enarr0w/dscraft/issues/18) — "[clean] Evaluate DeCoLe/Sanitizer feature parity vs. cleanlab's full surface"
- **Scope:** Effort 0.5, decision-only per the roadmap. Audit only; implement directly only if a gap is clearly small and obviously correct.

## Why this audit exists

`dscraft.clean.label_errors` (DeCoLe) is a from-scratch NumPy/SciPy implementation of group-conditioned
Confident Learning, built specifically so `dscraft.clean` never depends on `cleanlab` (AGPL-3.0) — see
CLAUDE.md's LazyIsolate licensing policy: `dscraft.clean`'s label-error detection *is* the network-facing-adjacent
service, so no subprocess-isolation workaround applies to an AGPL dependency here, full replacement is the only
option. `ai-helpdesk-agent`, a real local project, currently depends on `cleanlab` directly for label-error
detection. This audit asks: is DeCoLe's from-scratch surface actually close enough to cleanlab's real, commonly-used
API that a project like `ai-helpdesk-agent` could migrate off `cleanlab` onto `dscraft.clean` today — and if not,
exactly what's missing?

**Method note (licensing-compliant):** this audit compares against cleanlab's *published documentation* (its
`docs.cleanlab.ai` API reference and tutorials) and its *observable public API signatures* — never its source code.
Reading cleanlab's docs to describe its documented behavior does not create a derivative work; copying its
AGPL-licensed implementation would. No `cleanlab` source was read or copied in producing this audit. Doc sources
used: `cleanlab.filter.find_label_issues` API reference (docs.cleanlab.ai, `filter.py` reference page),
`cleanlab.multilabel_classification.rank.get_label_quality_scores`, `cleanlab.regression.rank.get_label_quality_scores`,
and the "Workflows of Data-Centric AI" tutorial.

## What `dscraft.clean` has today

| File | Capability |
|---|---|
| `label_errors.py` | `compute_group_class_thresholds`, `build_group_confident_joints`, `prune_by_noise_rate`, `prune_by_class`, `prune_by_both` (new, this pass), `detect_label_errors(strategy=...)` — group-conditioned Confident Learning over caller-supplied out-of-sample `probs`. |
| `contamination.py` | Two-stage LSHBloom (MinHash + banded Bloom filters) + Min-K%++ train/test contamination detection — not a cleanlab capability at all (cleanlab has no train/test contamination detector); orthogonal, not compared here. |
| `integrity.py` | Aggregate Dataset Integrity Score combining label-error, contamination, and demographic-drift signals — also not a cleanlab capability; DSCraft's own composition layer. |
| `__init__.py` (`Sanitizer`/`SanitizerReport`) | Composes the above three into one `audit()` → `purge()` workflow, including a "demographic-preserving" bounded per-group removal-rate purge strategy — this composed, group-aware workflow has no cleanlab equivalent at all (cleanlab has no `Sanitizer`-style purge with per-group removal caps). |

## Capability-by-capability comparison

| Capability | cleanlab (`cleanlab.filter.find_label_issues` et al.) | `dscraft.clean` | Verdict |
|---|---|---|---|
| Confident Learning core algorithm (confident joint, thresholded self-confidence) | Yes — the foundational algorithm | Yes, `build_group_confident_joints`/`compute_group_class_thresholds` implement the same public algorithm from scratch | **Parity** |
| Group/demographic-conditioned thresholds & confident joints | No — cleanlab's public API is single global threshold per class only; no first-class "group" argument in `find_label_issues` | Yes — this is DeCoLe's entire raison d'être (module docstring) | **`dscraft.clean` ahead** — a real capability cleanlab does not have |
| `filter_by="prune_by_noise_rate"` | Yes (cleanlab's *default*) | Yes — `prune_by_noise_rate` / `strategy="noise_rate"` (also this module's default) | **Parity** |
| `filter_by="prune_by_class"` | Yes | Yes — `prune_by_class` / `strategy="class"` | **Parity** (ranking criterion differs slightly: cleanlab ranks "smallest probability of belonging to given class" per class; DeCoLe ranks "own-label confidence minus own group/class typical confidence" — same intent, group-conditioned instead of global) |
| `filter_by="both"` (intersection of the two above) | Yes | **Was missing — now implemented** (`prune_by_both` / `strategy="both"`, this pass) | **Closed this pass** — see "Gap closed" below |
| `filter_by="confident_learning"` (flag every off-diagonal confident-joint cell, unranked/untruncated by `frac`) | Yes | No — `prune_by_noise_rate`/`prune_by_class` always rank-and-truncate to a `frac`/`n` budget; there is no "just give me every off-diagonal assignment" mode | **Gap** — small, NumPy-only, see recommendation below |
| `filter_by="predicted_neq_given"` (`argmax(probs) != observed_label`, no confident joint at all) | Yes | No | **Gap** — trivial (no confident joint needed), see recommendation below |
| `return_indices_ranked_by` (rank flagged examples by a continuous quality score instead of returning only a boolean mask) | Yes (`normalized_margin`, `self_confidence`, `confidence_weighted_entropy`) | No — every `prune_*` function returns only a boolean mask; the underlying scores (`assigned_class_joint_prob`, `own_label_prob - own_label_threshold`) exist internally in `GroupConfidentJoint` but are never surfaced as a public per-example score | **Gap** — real, moderate-value gap, see recommendation below |
| Aggregate per-example "label quality score" (`get_label_quality_scores`, continuous 0-1, not just a binary flag) | Yes, first-class (`cleanlab.rank.get_label_quality_scores`) | No public equivalent (see above — closely related to the `return_indices_ranked_by` gap; would likely be solved by the same underlying change) | **Gap**, same root cause as above |
| Multi-label classification support (`multi_label=True`) | Yes, dedicated API surface (`cleanlab.multilabel_classification`) | No | **Deliberately out of scope for this pass** — real feature, but non-trivial: DeCoLe's whole design (per-group, per-class thresholds and one confident joint per group) assumes single-label multiclass; extending to multi-label means redesigning the confident-joint construction (one-vs-rest per label, plus an aggregation rule across labels), not a NumPy-only bolt-on. Worth a dedicated future design pass, not a quick parity fix. |
| Regression label-error detection (`cleanlab.regression`) | Yes, dedicated module | No | **Deliberately out of scope for this pass** — a fundamentally different algorithm (residual/uncertainty-based, not confident-joint-based; cleanlab's regression path internally fits multiple models via cross-validation/bootstrapping to get prediction uncertainty). Bolting this onto `label_errors.py` would not be "extending DeCoLe," it would be a new module. Also has no natural group-conditioning story yet. Not recommended as a near-term follow-up. |
| `Datalab` (multi-issue-type audit: label issues + outliers + near-duplicates + non-IID + ... in one call) | Yes | Partially, at a different layer — `Sanitizer.audit()` already composes label-error detection + contamination + a dataset integrity score, which is `dscraft.clean`'s own (differently-scoped) multi-issue-type composition | **Not a real gap** — `dscraft.clean` already has a composed multi-signal audit entrypoint; it composes different signals (contamination instead of outlier/non-IID detection) by design, not by omission. Extending `Datalab`-style issue types (e.g. outlier/near-duplicate detection as first-class `Sanitizer` issue types beyond text near-dup) is a separate, larger scoping question outside this issue. |
| `CleanLearning` (drop-in `sklearn`-compatible wrapper that trains a model, computes out-of-sample `probs` via cross-validation, and calls `find_label_issues` for you) | Yes | No — DeCoLe always requires the caller to supply `probs` themselves | **Deliberately out of scope** — `dscraft.clean` is contractually PyTorch-free and does not own or wrap any classifier-training code path (that belongs to `dscraft.automl`, per the "no formal inter-module data contracts yet" / "subpackages never call into each other" rule in CLAUDE.md). Requiring the caller to supply `probs` is a deliberate, already-documented design choice (`label_errors.py` module docstring: "this module only does array math over already-computed out-of-sample predicted probabilities... which callers are expected to have produced themselves"), not an oversight. A `CleanLearning`-equivalent belongs in `dscraft.automl` if anywhere, wired to DeCoLe via the caller's own code, not via a new cross-module import. |
| PyTorch-based confidence estimators / deep-learning-specific tutorials (e.g. cleanlab's image/text deep-learning workflows) | Yes (cleanlab ships PyTorch-based examples/wrappers for image and text classifiers) | Architecturally forbidden | **Permanently out of scope** — `dscraft.clean` is contractually PyTorch-free (CLAUDE.md's shared-architecture section: ONNX Runtime only, <100MB target). Any deep-learning confidence-estimator wrapper would violate that constraint outright; this is not treated as a gap to close, ever. |

## Gap closed this pass

**`filter_by="both"` → `prune_by_both()` / `detect_label_errors(strategy="both")`.** This was exactly the
"clearly small, obviously-correct" case flagged in the issue: cleanlab's `"both"` mode is documented as simply
"filters only those examples that would be filtered by both `prune_by_noise_rate` and `prune_by_class`" — i.e. the
intersection of two masks DeCoLe already computes independently. No new algorithm, no new confident-joint logic;
`prune_by_both` is a two-line composition of the two existing pruning functions (`packages/dscraft/src/dscraft/clean/label_errors.py`).
Implemented with the same `frac`/`n` budget semantics as the two underlying strategies, and wired into
`detect_label_errors`'s `strategy` parameter alongside `"noise_rate"`/`"class"`. Tests added in
`packages/dscraft/tests/clean/test_label_errors.py`: `test_prune_by_both_is_the_intersection_of_noise_rate_and_class`,
`test_prune_by_both_can_be_empty_even_when_class_alone_flags_examples` (demonstrates the strictness property —
`"both"` can flag *nothing* even when one underlying strategy alone flags several examples), `test_prune_by_both_rejects_invalid_frac_and_n`,
and `test_detect_label_errors_both_strategy`.

## Gaps worth closing next (recommended, filed as follow-up issues)

1. **`filter_by="predicted_neq_given"` and `filter_by="confident_learning"` modes.** Both are small, NumPy-only,
   and use logic already present or trivially derivable in `label_errors.py`:
   - `predicted_neq_given` needs no confident joint at all — it's simply `np.argmax(probs, axis=1) != observed_label`,
     optionally still ranked/truncated by `frac`/`n` for API consistency with the other strategies.
   - `confident_learning` is "every off-diagonal confident-joint assignment, unranked" — i.e.
     `prune_by_noise_rate`'s existing candidate-selection logic (`assigned_class != -1 and assigned_class !=
     observed_label`, already computed per `GroupConfidentJoint`) with the top-`frac`/`n` truncation step skipped
     entirely (flag every candidate that exists, not just the top-scoring subset).
   These are lower priority than the score-surfacing gap below because they're narrower filter variants rather than
   new information, but they are genuinely one-function-each additions once someone sits down to do them. **Filed as
   a new, small follow-up issue** (see below) rather than implemented speculatively in this pass, since the
   `prune_by_both` gap already used up this issue's one clearly-obvious quick win and two more strategies in the same
   pass risks under-testing edge cases (e.g. `predicted_neq_given`'s interaction with the categorical-label encoding
   in `_validate_and_encode`) rather than doing each properly.

2. **Surface a continuous per-example label-quality score, not just a boolean mask.** This is the highest-value real
   gap. cleanlab's `get_label_quality_scores`/`return_indices_ranked_by` let a caller triage by *degree* of
   suspicion, not just a binary in/out decision at a fixed `frac`. `GroupConfidentJoint` already computes exactly the
   raw ingredients needed (`assigned_class_joint_prob`, `own_label_prob`, `own_label_threshold`) — every `prune_*`
   function throws this information away after using it to rank internally. A `label_quality_scores(joints,
   n_samples, *, method=...)` function returning a `(n_samples,)` float array (with a documented convention for
   examples that were never scoreable, e.g. `NaN` for a genuinely undefined class threshold, mirroring
   `GroupClassThresholds`'s existing NaN-fallback convention) would let `Sanitizer.purge()` also rank by score rather
   than only by `own_label_confidence` (which it already partially does — see `_own_label_confidence` in
   `__init__.py` — but that is a private, `label_errors`-independent re-derivation, not this module's own
   public score). **Filed as a second follow-up issue** (see below) — this is more than a one-function change (needs
   a documented scoring convention across both pruning strategies' differing score semantics) so it does not meet
   this issue's "implement directly if clearly small" bar, but it's concrete and scoped enough to be its own issue.

## Follow-up issues filed

- [`gr3enarr0w/dscraft#43`](https://github.com/gr3enarr0w/dscraft/issues/43) — Add `predicted_neq_given` and
  `confident_learning` `filter_by`/`strategy` modes to `dscraft.clean.label_errors.detect_label_errors`.
- [`gr3enarr0w/dscraft#44`](https://github.com/gr3enarr0w/dscraft/issues/44) — Add a public, continuous per-example
  label-quality score to `dscraft.clean.label_errors` (`label_quality_scores()`), surfacing the ranking signal
  `GroupConfidentJoint` already computes internally.

## Explicitly out of scope (not recommended, ever or for now)

- Multi-label classification support — real feature, but requires redesigning DeCoLe's confident-joint construction
  for one-vs-rest labels; not a parity quick-fix.
- Regression label-error detection — a different algorithm family entirely (uncertainty/residual-based, not
  confident-joint-based); would be a new module, not an extension of `label_errors.py`.
- `CleanLearning`-style train-a-model-for-you wrapper — would require `dscraft.clean` to own classifier training,
  which belongs to `dscraft.automl` per the "subpackages never call into each other" / "no inter-module contracts
  yet" rule in CLAUDE.md.
- Any PyTorch-based confidence estimator or deep-learning-specific workflow — forbidden outright by `dscraft.clean`'s
  ONNX-Runtime-only, PyTorch-free constraint (CLAUDE.md, shared-architecture section). Not treated as a gap; treated
  as a permanent non-goal.

## Bottom line for `ai-helpdesk-agent`

For the single-label, non-regression, non-multi-label case `ai-helpdesk-agent` is realistically exercising, DeCoLe
now covers all three of cleanlab's core `filter_by` ranking strategies (`prune_by_noise_rate`, `prune_by_class`,
`both`) plus a real capability cleanlab lacks entirely (group-conditioned thresholds). The remaining gaps
(`predicted_neq_given`/`confident_learning` filter variants, a continuous quality score) are real but narrow, and
neither blocks a migration off `cleanlab` today — they would only improve triage ergonomics once migrated. The two
categorically-different capabilities (multi-label, regression) are legitimate reasons `ai-helpdesk-agent` might
still need `cleanlab` if it actually exercises them; if it does not (this audit did not have access to
`ai-helpdesk-agent`'s source to confirm which `cleanlab` calls it makes), migration is realistic today.
