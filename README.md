# Benchcraft

A unified, MIT-licensed, local-first ML tooling platform spanning data curation, tabular AutoML, time-series forecasting, graph ML, computer vision, LLM fine-tuning, LLM/agent red-teaming, and agent/RAG benchmark evaluation.

See [`Benchcraft_Unified_Architecture.md`](./Benchcraft_Unified_Architecture.md) for the full locked v1 architecture spec, and [`CLAUDE.md`](./CLAUDE.md) for repo conventions and the module dependency graph.

## Packages

Each module is an independently-versioned Python package under `packages/`:

| Package | Status |
|---|---|
| `packages/lazycore` | Shared substrate (data-tier conventions, telemetry, licensing policy, sandbox executor) |
| `packages/automl` | Tabular AutoML |
| `packages/lazyclean` | Data-quality / deduplication |
| `packages/lazyforecast` | Time-series forecasting |
| `packages/lazygraph` | Graph ML |
| `packages/lazyvision` | Computer vision |
| `packages/lazytune` | LLM fine-tuning |
| `packages/lazyred` | LLM/agent red-teaming |
| `packages/lazyagent` | Agent/RAG benchmark evaluation |

## Local development install

Each package is independently pip-installable. To install every package in editable mode for local development:

```bash
pip install -r requirements-dev.txt
```

Or install a single package:

```bash
pip install -e packages/lazycore
pip install -e "packages/automl[onnx,dev]"
```

## Contributing

All changes land via pull request — direct pushes to `main` are disabled. Every PR is reviewed by CodeRabbit before merge.
