# DSCraft

A unified, MIT-licensed, local-first ML tooling platform spanning data curation, tabular AutoML, exploratory data analysis, time-series forecasting, graph ML, computer vision, LLM fine-tuning, LLM/agent red-teaming, and agent/RAG benchmark evaluation.

See [`DSCraft_Unified_Architecture.md`](./DSCraft_Unified_Architecture.md) for the full locked v1 architecture spec, and [`CLAUDE.md`](./CLAUDE.md) for repo conventions and the module dependency graph.

## One real package, `dscraft`

DSCraft ships as a single installable Python package — the way numpy or
PyTorch ships one distribution with an internal module tree, not as
separately-installed packages per capability. Each capability lives in its
own subpackage, and each subpackage's heavy dependencies are gated behind
its own pip extra, so installing one subpackage never forces installing
another's conflicting dependency stack.

```bash
pip install dscraft                  # base install: just dscraft.core
pip install "dscraft[automl]"        # + dscraft.automl's runtime deps
pip install "dscraft[all]"           # every subpackage's runtime deps
```

```python
import dscraft.core        # always available
import dscraft.automl      # available once installed with the `automl` extra
```

| Subpackage | Extra | What it does |
|---|---|---|
| `dscraft.core` | *(base install)* | Shared substrate: three-tier data conventions, OTel GenAI telemetry helpers, license-isolation policy, shared sandbox executor. |
| `dscraft.automl` | `automl` (+ `automl-onnx`) | Tabular AutoML — `.compile()` fuses a fitted `sklearn` pipeline into one portable ONNX graph. |
| `dscraft.clean` | `clean` | Data-quality firewall — ONNX Runtime (PyTorch-free) embeddings feeding near-duplicate/label-error detection. |
| `dscraft.eda` | `eda` | Exploratory data analysis — a lazy Polars profiling engine, HLL/KLL sketches, a mixed-type association matrix, and a self-contained HTML/Canvas report. |
| `dscraft.forecast` | `forecast` | Time-series forecasting — classical statistical models over an Arrow-backed pipeline. |
| `dscraft.graph` | `graph` | Graph ML — a sparse COO/CSR-CSC tensor adapter plus MPNNs. |
| `dscraft.vision` | `vision` | Computer vision — a dense image pipeline plus CNN/ViT export via `torch.export()`→ONNX. |
| `dscraft.tune` | `tune` | Local LLM fine-tuning — an adapter-factory interface over LoRA/`peft`. |
| `dscraft.security` | `security` | LLM red-teaming — sandboxed probe/detector loops against local targets, OWASP-mapped findings. |
| `dscraft.agent` | `agent` | Agent/RAG benchmark evaluation — a bring-your-own-agent adapter scored inside the shared sandbox. |

See [`packages/dscraft/README.md`](./packages/dscraft/README.md) for the
full install matrix, per-subpackage detail, and local-development
instructions.

## Local development

```bash
pip install -e "packages/dscraft[dev,all]"
pytest packages/dscraft/tests
```

## Contributing

All changes land via pull request — direct pushes to `main` are disabled. Every PR is reviewed by CodeRabbit before merge. See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the full workflow.
