# DSCraft Expansion Roadmap

This roadmap prioritizes the 32 planning issues (#7–#38) filed against `gr3enarr0w/dscraft` from a two-machine DS/ML tooling audit and a review of the open textbook *Forecasting: Principles and Practice, the Pythonic Way* (Hyndman et al., OTexts.com/fpppy). Every issue is planning-only — none of this has been implemented yet. This document exists to hand off a defensible execution order, not to guess at one.

## Methodology: RICE

Each issue is scored with **RICE**: `(Reach × Impact × Confidence) ÷ Effort`. This is a standard prioritization framework (used at Intercom and widely elsewhere) chosen specifically to avoid ranking-by-gut-feel — every score below is derived from evidence already gathered in the audit, not intuition.

- **Reach (1–5)**: how many independent real evidence sources (real local projects, textbook chapters, or other filed issues that depend on this landing first) support this issue. 5 = cross-cutting/affects most of the roadmap; 1 = single evidence source.
- **Impact (0.5–5)**: value delivered where it lands. 5 = flagship, currently-nonexistent capability a whole subpackage's stated purpose depends on; 3 = high; 2 = medium; 1 = low; 0.5 = minimal (e.g. cosmetic).
- **Confidence (50%/65%/80%/100%)**: how solid the Reach/Impact estimate is. 100% = real, repeated, or currently-active evidence with no open technical risk. 80% = solid but narrower evidence. 65% = single-source evidence or genuine open technical risk (e.g. unverified MPS support, a build-toolchain dependency). 50% = speculative (not used below — everything here has at least single-source real evidence).
- **Effort (0.5/1/2/3/4/6, relative units)**: 0.5 = pure evaluation/decision (no implementation), 1 = small self-contained PR, 2 = medium (new dependency + real integration work), 3 = medium-large, 4 = large (new subsystem), 6 = very large.

### A known, deliberate limitation of RICE — read before treating this as a strict sequence

RICE divides by effort, so it systematically favors small, low-risk wins over large strategic bets — even when those bets are more important. **#12 (RAG pipeline) and #13 (benchmark-adapter layer)** are the clearest example: both score maximum Reach (5), Impact (5), and Confidence (100%) — the strongest evidence and highest stakes of anything in this batch — but land mid-table because their effort is large. That is RICE working as designed, not a signal to deprioritize `agent`. Recommended reading: use the RICE order below as the default sequence for everything *except* #12/#13, and start those two in parallel as a longer-running track from early on, since `dscraft.agent`'s entire reason for existing depends on them and they'll take the longest regardless of when they start.

## Ranked issues

| Rank | RICE | Issue | R | I | C | E | Why (evidence) |
|---|---|---|---|---|---|---|---|
| 1 | 24.00 | [#20 forecast: broaden StatsForecast allowlist](https://github.com/gr3enarr0w/dscraft/issues/20) | 4 | 3 | 100% | 0.5 | Zero new dependencies — every model (Naive/SeasonalNaive/Drift/MSTL/AutoTheta/AutoCES/Croston) already ships in the existing `statsforecast` dependency. Pure allowlist expansion. |
| 2 | 12.00 | [#14 core: shared embeddings helper evaluation](https://github.com/gr3enarr0w/dscraft/issues/14) | 3 | 2 | 100% | 0.5 | Decision-only. Directly unblocks #12 and keeps `clean`'s PyTorch-free promise from being violated by accident. |
| 3 | 10.00 | [#29 forecast: adopt utilsforecast](https://github.com/gr3enarr0w/dscraft/issues/29) | 5 | 2 | 100% | 1 | Appears across nearly every book chapter; is shared infra for #21/#24/#26/#27. Landing this first avoids four separate reworks later. |
| 3 | 10.00 | [#38 core: cross-extra dependency-version compatibility](https://github.com/gr3enarr0w/dscraft/issues/38) | 5 | 3 | 100% | 1.5 | Torch already has 3 different version floors across `vision`/`graph`/`tune` today — this is an observable, growing risk, not speculative. Affects every other issue's installability. |
| 5 | 9.00 | [#7 automl: XGBoost/LightGBM/CatBoost backends](https://github.com/gr3enarr0w/dscraft/issues/7) | 3 | 3 | 100% | 1 | 2 independent real projects (`himalaya_email_classifier`, `ai-helpdesk-agent`); well-trodden extension of the existing `SUPPORTED_MODELS` pattern. |
| 6 | 6.25 | [#12 agent: RAG pipeline layer](https://github.com/gr3enarr0w/dscraft/issues/12) | 5 | 5 | 100% | 4 | 5 independent real projects across both machines (`hermes-rag-qdrant-efficient-skill`, `Locomo-Plus`, `database_vector_creation`, `vector_n8n_work`, `qdrant_storage`). See the RICE caveat above. |
| 6 | 6.25 | [#13 agent: benchmark-adapter layer](https://github.com/gr3enarr0w/dscraft/issues/13) | 5 | 5 | 100% | 4 | 6 independent real benchmark harnesses reimplement the same pattern this generalizes (`AgentBench`, `SWE-bench`, `tau2-bench`, `LongMemEval-V2`, `inspect-metr-task-bridge`, `Locomo-Plus`). See the RICE caveat above. |
| 8 | 6.00 | [#21 forecast: MLForecast (tree-based) backend](https://github.com/gr3enarr0w/dscraft/issues/21) | 4 | 3 | 100% | 2 | Already committed to in the architecture doc; confirmed again by the book (ch. 7, 10). |
| 8 | 6.00 | [#22 forecast: decomposition (STL/MSTL/Box-Cox)](https://github.com/gr3enarr0w/dscraft/issues/22) | 3 | 2 | 100% | 1 | Zero new dependencies (`statsmodels` already present); feeds #23 and MSTL forecasting in #20. |
| 8 | 6.00 | [#26 forecast: conformal prediction intervals](https://github.com/gr3enarr0w/dscraft/issues/26) | 4 | 3 | 100% | 2 | Already committed to in the architecture doc; benefits every model, not one algorithm family. |
| 11 | 5.00 | [#19 docs: end-to-end notebook examples](https://github.com/gr3enarr0w/dscraft/issues/19) | 5 | 1 | 100% | 1 | Nearly every real project found across both audits is notebook-first. Low impact per unit (doesn't add capability) but very broad reach and trivial effort. |
| 12 | 4.00 | [#18 clean: cleanlab parity audit](https://github.com/gr3enarr0w/dscraft/issues/18) | 1 | 2 | 100% | 0.5 | Audit-only against a known, real AGPL dependency (`ai-helpdesk-agent`). Clear scope, real licensing-risk mitigation value. |
| 12 | 4.00 | [#31 forecast: adopt datasetsforecast (M3/M4/M5)](https://github.com/gr3enarr0w/dscraft/issues/31) | 2 | 2 | 100% | 1 | Test-only dependency; validated by a real local project (`2026_time_series_m4_final`) already doing exactly this comparison. |
| 14 | 3.20 | [#11 vision: OCR (EasyOCR/Tesseract)](https://github.com/gr3enarr0w/dscraft/issues/11) | 2 | 2 | 80% | 1 | 2 independent real projects; mature libraries, narrow bolt-on scope. |
| 15 | 3.00 | [#25 forecast: zero-shot foundation models](https://github.com/gr3enarr0w/dscraft/issues/25) | 4 | 3 | 100% | 4 | Book ch. 15 + a real, independently-built local project (`2026_time_series_m4_final`) already doing this comparison. High confidence, high effort. |
| 15 | 3.00 | [#28 forecast: dynamic regression + VAR](https://github.com/gr3enarr0w/dscraft/issues/28) | 3 | 2 | 100% | 2 | Zero new dependencies (`statsmodels`); real coursework evidence (`assignment_6_ts`) plus book ch. 10/12. |
| 17 | 2.60 | [#32 new-capability: recommender systems evaluation](https://github.com/gr3enarr0w/dscraft/issues/32) | 1 | 2 | 65% | 0.5 | Single project (`mlii-music-rec`) but a very deep real stack. Evaluation-only, cheap to close. |
| 17 | 2.60 | [#33 clean/vision: image dedup evaluation](https://github.com/gr3enarr0w/dscraft/issues/33) | 1 | 2 | 65% | 0.5 | Single project (`photo_dedupe_project`); resolves a real architectural tension (PyTorch-free `clean` vs. CLIP). |
| 17 | 2.60 | [#35 core: experiment tracking evaluation](https://github.com/gr3enarr0w/dscraft/issues/35) | 1 | 2 | 65% | 0.5 | Single project (`mlii-music-rec`, MLflow+W&B together). Evaluation-only. |
| 20 | 1.95 | [#8 automl: Optuna HPO backend](https://github.com/gr3enarr0w/dscraft/issues/8) | 3 | 2 | 65% | 2 | 2 real projects; confidence lowered by the open Ray Tune vs. Optuna multi-backend question raised on #24. |
| 21 | 1.60 | [#9 automl: HDBSCAN + imbalanced-learn](https://github.com/gr3enarr0w/dscraft/issues/9) | 1 | 2 | 80% | 1 | Single project (`ai-helpdesk-agent`) but both techniques are clearly scoped and low-risk. |
| 22 | 1.50 | [#34 tune: synthetic training-data generation](https://github.com/gr3enarr0w/dscraft/issues/34) | 2 | 3 | 100% | 4 | A currently-active real production pipeline (`Running_AI` + root CLAUDE.md, 120k examples). High confidence, but large effort and requires careful no-router guardrails. |
| 23 | 1.30 | [#15 eda: UMAP/PCA/t-SNE projection](https://github.com/gr3enarr0w/dscraft/issues/15) | 1 | 2 | 65% | 1 | Single project (`hermes-rag-qdrant-efficient-skill`) for UMAP specifically; PCA is free via existing deps. |
| 23 | 1.30 | [#16 eda/core: warehouse source connectors](https://github.com/gr3enarr0w/dscraft/issues/16) | 2 | 2 | 65% | 2 | Real recurring professional workflow (`jira-dbt`, `jsm-modeling`) but connector maturity/testing without live warehouse access is a real open question. |
| 23 | 1.30 | [#23 forecast: tsfeatures-style feature extraction](https://github.com/gr3enarr0w/dscraft/issues/23) | 2 | 2 | 65% | 2 | Single book source (ch. 4); must also stay carefully decoupled from `eda`. |
| 26 | 0.67 | [#37 automl: AutoGluon backend option](https://github.com/gr3enarr0w/dscraft/issues/37) | 1 | 2 | 100% | 3 | Real evidence (`2026_time_series_m4_final`), no licensing ambiguity (Apache-2.0), but explicitly "additional, not replacement" — caps its incremental value. |
| 27 | 0.65 | [#10 vision: TensorFlow/Keras backend](https://github.com/gr3enarr0w/dscraft/issues/10) | 2 | 2 | 65% | 4 | Real coursework evidence, but MPS/`tensorflow-metal` support is an unverified risk and the effort (parallel backend, dual preprocessing) is large. |
| 27 | 0.65 | [#17 eda: plotnine (grammar-of-graphics)](https://github.com/gr3enarr0w/dscraft/issues/17) | 1 | 1 | 65% | 1 | Single project (`wine-quality-eda`, in R — this is the Python-equivalent substitute, not a direct evidenced Python need). Cosmetic/API-breadth value. |
| 27 | 0.65 | [#24 forecast: NeuralForecast (deep learning) backend](https://github.com/gr3enarr0w/dscraft/issues/24) | 2 | 2 | 65% | 4 | Extensive book coverage (ch. 14) but no real-project confirmation specific to time-series deep learning in either audit; MPS/PyTorch-Lightning support unverified. |
| 27 | 0.65 | [#27 forecast: hierarchical/grouped reconciliation](https://github.com/gr3enarr0w/dscraft/issues/27) | 1 | 2 | 65% | 2 | Single book source (ch. 11), no real local usage found in either audit. |
| 27 | 0.65 | [#30 forecast: Prophet backend](https://github.com/gr3enarr0w/dscraft/issues/30) | 1 | 2 | 65% | 2 | Single book source (ch. 12), no real local usage found; Stan/`cmdstanpy` build friction on Apple Silicon is a real, unverified risk. |
| 32 | 0.40 | [#36 forecast: GARCH/volatility modeling](https://github.com/gr3enarr0w/dscraft/issues/36) | 1 | 1 | 80% | 2 | Single project (`assignment_7_ts`), narrowest addressable audience (financial time series specifically) of anything in the forecast batch. |

## How to use this on handoff

1. Work the table top-down as the default sequence — it's the objective, evidence-derived order, not a suggestion to second-guess per-issue.
2. Start #12 and #13 (`agent`) as a parallel, longer-running track early rather than waiting for their literal rank — see the RICE caveat above. They anchor the entire `agent` subpackage's stated purpose.
3. Every issue is independently scoped and cites its real evidence in its GitHub issue body — read the issue itself before starting, not just this table.
4. Ties (same RICE score) have no implied order between them; sequence by whichever fits current context (e.g. don't context-switch dependency stacks mid-batch).
5. This ranking reflects evidence gathered on 2026-07-20 from two machines' `~/Development` folders plus the *fpppy* textbook. If new real usage is found later (e.g. a third machine), re-score rather than append unscored issues to the bottom.
