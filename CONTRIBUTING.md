# Contributing to DSCraft

This document describes the actual, currently-configured contribution and
review workflow for this repository — verified against the live GitHub
branch protection settings, `.coderabbit.yaml`, and
`.github/workflows/test.yml` at the time of writing. It is process-focused:
for architecture, module conventions, and coding rules (packaging layout,
"no net-new scripts," subagent workflow, etc.), see `CLAUDE.md`.

## 1. Local development setup

There is one package, `dscraft`, install it in editable mode with every
subpackage's runtime and test dependencies:

```bash
pip install -e "packages/dscraft[dev,all]"
```

Run the full suite:

```bash
pytest packages/dscraft/tests
```

Or install/test just the subpackage extras you're working on (e.g. to run
only `forecast`'s tests without pulling in `torch`):

```bash
pip install -e "packages/dscraft[forecast,dev]"
pytest packages/dscraft/tests/forecast
```

See `CLAUDE.md`'s "Repository status" section for the full subpackage list
and the module dependency graph.

## 2. Branch protection on `main`

`main` is protected. Verified via `gh api repos/gr3enarr0w/dscraft/branches/main/protection`:

- **No direct pushes.** All changes must go through a pull request.
- **1 approving review required** (`required_approving_review_count: 1`).
- **Stale reviews are dismissed automatically** when new commits are pushed
  to the PR branch (`dismiss_stale_reviews: true`) — an approval only covers
  the commit it was given on; pushing a fix round requires a fresh approval.
- **Force pushes and branch deletion are disabled** on `main`
  (`allow_force_pushes: false`, `allow_deletions: false`).
- **`enforce_admins` is on** — repo admins are not exempt from any of the
  above.
- **Required status checks** (target state — see note below):
  - `CodeRabbit`
  - `test (core)`
  - `test (automl)`
  - `test (clean)`
  - `test (eda)`
  - `test (forecast)`
  - `test (graph)`
  - `test (vision)`
  - `test (tune)`
  - `test (security)`
  - `test (agent)`
  - `examples syntax check`

  **This list describes the intended/target state, not what's live on
  GitHub yet.** The check names above assume `.github/workflows/test.yml`'s
  matrix uses the new `dscraft` subpackage names (`core`, `automl`,
  `clean`, `eda`, `forecast`, `graph`, `vision`, `tune`, `security`,
  `agent`) as its `subpackage` matrix value, which is what determines each job's
  display name. The actual GitHub branch-protection required-status-checks
  configuration has **not** been updated to this list yet — that requires
  a `gh api` call with a valid token, which is a separate, later step
  gated on a fresh GitHub token being available. Until that step runs,
  the live required-checks list on GitHub may still reference the old
  nine-separate-package check names (`test (lazycore)`,
  `test (automl)`, `test (lazyclean)`, etc.). Don't assume the list above
  is already enforced; verify with the `gh api` command above before
  relying on it.

None of this is friction to route around — it's the intended safety net for
a solo/small-team repo where a single bad merge to `main` would break every
subpackage of the one `dscraft` package at once. If a check seems wrong for
a given PR, the fix is to fix the PR (or, for a CodeRabbit finding that
contradicts a locked architecture decision, to explain why in a PR comment
— see §5), not to bypass the check.

## 3. CI (`.github/workflows/test.yml`)

CI runs on every push to `main` and every pull request targeting `main`.
There are two jobs:

- **`test` matrix** — one job per `dscraft` subpackage, named `test
  (<subpackage>)`, each running a single `pip install -e
  "packages/dscraft[<extras>,dev]"` install step (no more two-step
  install-the-base-package-first-then-the-module dance — there's only one
  package now) followed by `pytest packages/dscraft/tests/<subpackage>`.
  The matrix covers all ten subpackages: `core`, `automl`, `clean`, `eda`,
  `forecast`, `graph`, `vision`, `tune`, `security`, `agent`.
  - `automl` installs the `automl,automl-onnx,dev` extras (its
    `.compile()`→ONNX export path is optional, not core).
  - **`security` and `agent` run on `macos-14`, not `ubuntu-latest`.**
    This is deliberate, not an oversight: their test suites exercise the
    real `dscraft.core` sandbox executor (not a mock), and per `CLAUDE.md` /
    the architecture doc, the Linux sandbox backend is an explicit,
    documented, unimplemented stub that always raises
    `SandboxBackendUnavailableError`. Running these suites on
    `ubuntu-latest` wouldn't test "the same thing on Linux" — it would
    deterministically fail every time regardless of any real regression.
    `macos-14` gives them a real Apple Silicon runner with actual Seatbelt
    support, matching this repo's stated primary reference platform.
  - The rest (`core`, `automl`, `clean`, `eda`, `forecast`, `graph`,
    `vision`, `tune`) run on `ubuntu-latest`.
- **`examples syntax check`** — runs `python -m py_compile` over every
  `packages/dscraft/examples/**/*.py` file. This only catches syntax
  errors/typos; it does not import or execute the examples, so it needs
  none of the subpackages' heavy runtime dependencies installed. It is a
  deliberately lightweight check, not a substitute for actually running an
  example.

CI exists because CodeRabbit (§4) reviews code quality and style — it does
not execute anything. Without this workflow, nothing automated verifies
that the test suites actually pass before a PR can merge.

## 4. CodeRabbit (`.coderabbit.yaml`)

CodeRabbit is a **required** status check on `main` (see §2) and also
auto-reviews **draft** PRs, not just PRs marked ready for review
(`reviews.auto_review.drafts: true`) — so a finding can show up before you
even open the PR for review.

### Review strictness by path

CodeRabbit applies different review instructions depending on what part of
a package a diff touches:

- **`packages/*/src/**`** (production library code) — reviewed strictly:
  missing/incorrect type hints, undocumented public functions/classes,
  silent exception swallowing, and any deviation from `CLAUDE.md`'s
  licensing policy (no GPL/AGPL runtime deps outside the documented
  subprocess-isolation pattern, no torch/transformers in `lazyclean`, no
  torch-sparse/METIS in `lazygraph`). It also flags any new shared
  inter-module interface, router, or `BaseTarget`-style abstraction as
  out of scope, per `CLAUDE.md`.
- **`packages/*/examples/**`** (runnable demos) — reviewed leniently:
  docstring coverage and exhaustive error handling are not required. It
  does still flag a demo that reimplements logic instead of importing it
  from the package's public API (`CLAUDE.md`'s "no net-new scripts" rule
  applies to examples too).
- **`packages/*/tests/**`** — reviewed for tests that don't actually
  assert anything, over-mocked tests that no longer exercise real
  behavior, and use of `pytest.importorskip` that would silently skip an
  entire tier of coverage.

### Pre-merge checks

- **Docstring coverage threshold: 85%**, mode `warning` (not `error`) —
  raised from CodeRabbit's 80% default because the repo already sits
  higher after a dedicated docstring pass. Left at `warning` rather than
  `error` so a docstring gap (a style issue) can't fail the required
  `CodeRabbit` status check the way an actual failing test would.
- **`Tests updated for src changes`** (custom check, `warning`) — flags a
  PR that adds/modifies `packages/*/src/**` without a corresponding change
  under that package's `packages/*/tests/**`, unless it's a
  comment/docstring-only change, a behavior-preserving rename, or the PR
  description explicitly justifies why no test change is needed.
- **`No scope-locked architecture violations`** (custom check, `warning`)
  — cross-references the diff against Part 4 ("Cross-Cutting Decisions
  Reference Table") of `DSCraft_Unified_Architecture.md` and fails,
  quoting the specific row(s) violated, if the diff does any of:
  1. Adds a shared LLM router, multi-provider LLM abstraction, or a
     `BaseTarget`/router-style base class meant to be shared across more
     than one module.
  2. Adds a new formal, typed inter-module data contract, shared manifest
     schema, or MLflow-"flavors"-style interface between two DSCraft
     modules before two real modules actually need to exchange data.
  3. Adds code under `packages/dscraft/src/dscraft/core/` that implements
     logic specific to one non-`core` subpackage, rather than a shared
     data-tier/OTel-schema/license-policy/sandbox-executor convention.
  4. Adds a cloud deployment target, a remote/hosted inference endpoint,
     or a non-local execution path for any module, including
     LazyRed/LazyAgent targets.
  5. Adds any implementation code (not comments/docs) for LazyEdge
     (edge/microcontroller compilation), which is fully deferred.

  If `DSCraft_Unified_Architecture.md` isn't present in the diff or
  repo context, this check states that explicitly and passes with a note
  rather than failing blind.

Both custom checks currently run in `warning` mode, so they surface as
CodeRabbit review comments rather than blocking the `CodeRabbit` status
check by themselves — but because `CodeRabbit` itself is a required status
check (§2), a PR with unresolved actionable findings from either check
should not be merged (see the fix-round workflow in §6).

## 5. Self-approval and the write-permission requirement

This is undocumented-elsewhere tribal knowledge from getting this workflow
running, and it will bite the next contributor if it isn't written down:

**GitHub will not let a PR author approve their own PR, and it fails
silently.** There is no error message — the "Approve" review submits
successfully, but GitHub's branch-protection review count simply does not
increment for it.

In this repo, PRs are authored under the `gr3enarr0w` account, so approval
must come from a different account. But a different *identity* is not
sufficient on its own:

> **The approving account must also have at least `push` (write)
> permission on the repository.** An account with only read access can
> submit an "Approve" review through the UI or API, and GitHub will accept
> it without error — but it does **not** count toward the
> `required_approving_review_count` in branch protection. This failure
> mode is also silent: the review shows as "Approved" in the PR UI, and
> nothing indicates it isn't satisfying the merge requirement.

### The fix

1. Add the approving account as a collaborator with write access:

   ```bash
   gh api repos/OWNER/REPO/collaborators/USERNAME -X PUT -f permission=push
   ```

2. The invited account must accept the invitation. Authenticated as that
   account:

   ```bash
   gh api -X PATCH user/repository_invitations/<invitation_id>
   ```

3. **A previously-submitted approval retroactively counts** once the
   account has write access — there is no need to dismiss and resubmit the
   review. If the approval was submitted while the account only had read
   access, upgrading the permission alone is enough to make the existing
   review satisfy branch protection.

If a PR looks "Approved" but still shows the merge button disabled or the
review requirement unmet, check the approving account's permission level
before assuming something else is wrong.

## 6. Merge method

**Squash and merge** is the convention for this repo. PRs typically go
through one or more CodeRabbit fix rounds (see §7), which produce several
incremental commits (e.g. "fix: address CodeRabbit review comments").
Squash-merging collapses all of that into a single clean commit on `main`,
so `main`'s history reads as one commit per logical change rather than one
commit per fix-round iteration.

**"Close pull request" is a separate, unrelated action** — it abandons the
PR without merging any of its changes into `main`. Do not confuse it with
a post-merge cleanup step; GitHub already closes a PR automatically when it
is merged. Only use "Close pull request" for a PR that should genuinely be
abandoned.

## 7. CodeRabbit fix-round workflow

This is the process actually used to take a PR from open to merged in this
repo. Follow it in order:

1. **Run a local structured code review before pushing or opening the
   PR.** Use this repo's `/code-review` skill, or an equivalent manual
   review pass, on the diff first. This is a hard requirement established
   during this project's early PRs — it is not optional, and it exists to
   catch what CodeRabbit will catch anyway, before spending a review round
   on it.
2. **Push and open the PR**, and let CodeRabbit run its automated review
   (it will also run against a draft PR — see §4).
3. **Triage every actionable CodeRabbit finding.** For each one, either:
   - **Fix it**, or
   - **Decline it with a citation**, if it contradicts a locked decision
     in `DSCraft_Unified_Architecture.md` Part 4 — leave a PR comment
     replying to the finding, explaining why it's being declined, and
     citing the specific locked row it conflicts with.

   Never silently ignore a finding either way — every actionable comment
   gets an explicit resolution (a fix commit or a reasoned decline).
4. **After fixing, verify independently before pushing again.** Do a
   fresh virtualenv install and run `pytest` yourself — don't just trust a
   subagent's or reviewer's self-report that a fix works.
5. **Repeat steps 2–4** until CodeRabbit's review shows zero actionable,
   open threads.

Only once CI is green, `CodeRabbit` has no open actionable threads, and an
approval from a qualifying account (§5) is in place should the PR be
squash-merged (§6).
