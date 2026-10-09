# Weekly logs

One file per week, named by the week's **end date**: `YYYY-MM-DD.md`. Each one
records what happened that week — status, submissions, results as they landed,
bugs found, and a dated narrative at the bottom.

The split:

- **Log** (here) — what happened, when, and what was decided. Append-only in
  spirit; a past week is not rewritten when later results change the picture.
- **[`../experiments/`](../experiments/)** — one page per experiment, kept
  current. The log links to it rather than repeating the detail.
- **[`../results/`](../results/)** — settled numbers, canonical.

Update the current week's file whenever a job starts or finishes, a result
lands, or a decision is made. Start a new file on the week boundary and carry
forward only what is still open.

| Week ending | Theme |
|---|---|
| [2026-10-13](2026-10-13.md) | Fixing what the analysis found: live history extractor (exp 4), proprio encoder without miscalibrated context (exp 5) |
| [2026-10-06](2026-10-06.md) | Conditioning week: C-token adaLN vs Direct delta_M on random history |
| [2026-09-29](2026-09-29.md) | Ablation week: what VideoPooled's robustness actually needs |
| [2026-09-22](2026-09-22.md) | Eval-run index: every campaign written that week |

New week: `cp TEMPLATE.md YYYY-MM-DD.md`.
