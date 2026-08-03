# Greptile configuration — the CodeRabbit fallback

Greptile is **not** this repo's primary reviewer. CodeRabbit is (CLAUDE.md §5.1 step 3).
Greptile exists here as the **fallback** for when CodeRabbit is rate- or quota-limited, and
it is constrained by two rules: it never fires on its own, and it is capped at **16 reviews
per calendar month**, anchored **2026-08-03**.

Greptile reads `config.json` and `rules.md` from this directory. This `README.md` is not
part of that schema — it is here so the config's rationale sits next to the config.

## What `config.json` pins, and why

| Key | Value | Why |
|---|---|---|
| `skipReview` | `"AUTOMATIC"` | The whole point: disables automatic PR review while leaving the manual `@greptileai` comment trigger working. Must be **exactly** `"AUTOMATIC"` — no other value is valid, and an invalid one is ignored, silently restoring auto-review. |
| `triggerOnUpdates` | `false` | Belt-and-braces. `true` re-reviews on **every push**, which would burn the monthly cap in a single PR. |
| `statusCheck` | `false` | Greptile will normally never run, so a status check it posts would be a check that never arrives. `main` is currently unprotected, so this cannot block a merge today — pinned so that enabling branch protection later can't turn it into one. |
| `statusCommentsEnabled` | `false` | No progress chatter on PRs it was not asked to review. |
| `fileChangeLimit` | `50` | **Load-bearing.** The Greptile dashboard has this set to **1**, which was observed refusing an ordinary 4-file PR outright (`Too many files changed for review. (4 files found, 1 file limit)`). 11 of the last 12 PRs here touch more than one file (median ~9, max 32), so at the dashboard's value Greptile could not review essentially anything and would be useless as a fallback. 50 covers the observed distribution with headroom. |
| `autoApprove.enabled` | `false` | A fallback reviewer must never approve a PR by itself; approval authority stays with the CLAUDE.md §5.1 gate. |
| `ignorePatterns` | see file | Mirrors `.coderabbit.yaml`'s `path_filters` so the fallback reviews the *same* scope as the primary — prose (`*.md`, `*.qmd`, ADRs), figures, data, covariance models and lockfiles are excluded. Also self-excludes `.greptile/` and `greptile.json`. |

Only `.greptile/config.json` is used; a root `greptile.json` would be **ignored** whenever a
`.greptile/` directory exists in the same directory, so there is deliberately no such file.

## Precedence — this file beats the dashboard

Greptile's documented precedence, lowest to highest:

```
Dashboard settings        ← base defaults from the UI
Org default rules         ← overridden by any repo-level config
Root .greptile/           ← THIS FILE
Intermediate .greptile/
Most specific .greptile/
Org enforced rules        ← cannot be overridden by any config
```

So this config overrides whatever the dashboard says. The one thing that could still force an
automatic review is an **org enforced rule** — and this repo is owned by an individual user
account, not an organisation, so there are none to enforce.

## Two known ways an unrequested review can still happen

Both are transition effects, not steady state. In a repo that has carried
`skipReview:"AUTOMATIC"` continuously, the observed automatic-review rate is zero
(NVIDIA/Megatron-LM: 0 automatic reviews across 911 PRs over two months).

1. **The PR that lands this config can itself be auto-reviewed.** The PR-level rule is
   conjunctive — Greptile skips *"only if all applicable configs specify `AUTOMATIC`"* — and
   while that PR is open the default branch does not yet contain this file. This is not
   hypothetical: NVIDIA/Megatron-LM#5166, the PR that *restored* their config, was
   automatically reviewed with zero `@greptileai` mentions on it.
2. **Removing or breaking this file re-enables automatic review immediately and silently.**
   Megatron-LM deleted theirs on 2026-06-02 and took 23 unwanted reviews in the 62 hours
   before it was restored. A malformed JSON edit here has the same effect as deleting it.

Both are *detected* rather than trusted: `scripts/greptile_budget.py` flags any Greptile
activity on a thread with no manual trigger as `auto_fire_suspected`, and charges it to the
monthly budget. Both were observed on the PR that introduced this config (#98): Greptile
auto-reviewed it, and the counter caught and charged it.

**This config only takes effect once it is on `main`.** Greptile reads it from the default
branch, not from a PR head — on #98 it kept applying the dashboard's `fileChangeLimit: 1`
and refused a 4-file PR even after the file existed on the branch. So expect dashboard
behaviour, not this file's behaviour, on any PR opened before it merged.

## The 16/month cap

Greptile has **no native per-repo review quota**. Its only usage control is an
organisation-wide *dollar* cap on flex/overage spend, which is neither per-repo nor a review
count. So the cap is enforced repo-side, by:

```bash
GH_TOKEN=$(gh auth token --user bioedca) python scripts/greptile_budget.py
#   exit 0 → budget available (prints how many remain)
#   exit 2 → exhausted; do NOT invoke Greptile this period
#   exit 3 → could not measure; do NOT invoke Greptile (fail-closed)
#   exit 4 → usage error (bad CLI args); nothing was measured
```

Only **0** means "invoke". Exit 4 exists because argparse's own usage-error exit is 2 — without
the remap, a mistyped flag would report as "this month's budget is spent".

Run it **before** commenting `@greptileai`. The count is re-derived from the GitHub API on
every call rather than read from a ledger this repo writes about itself, so an invocation that
went unlogged cannot inflate the remaining budget. See that script's module docstring for why
the *trigger* is the counting unit and Greptile's own output is not.

**Both trigger handles count.** `@greptileai` is the documented one; `@greptile-apps` is the
one Greptile itself offers when a PR trips `fileChangeLimit` ("Bypass the limit by tagging
`@greptile-apps` to review."). The counter matches both — counting only the documented handle
would let a real invocation go unbilled, which is permissive in exactly the direction that
overruns the cap.
