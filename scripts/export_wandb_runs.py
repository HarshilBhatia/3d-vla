#!/usr/bin/env python3
"""Export the 3DFA wandb runs (config, full metric history, summary) for archival.

Exports one directory per run under ``--out``::

    <out>/<run_name>__<run_id>/
        config.json     # run.config (resolved hyperparameters)
        summary.json     # run.summary (final/best scalar values)
        meta.json        # run name/id/state/created_at/tags/url/runtime
        history.csv      # full metric history via scan_history (all steps, no downsampling)
        history.parquet  # same, if pyarrow is available

``run.history()`` downsamples to ``samples`` rows; ``scan_history()`` streams every
logged step, which is what an archive needs. Auth comes from the environment
(``WANDB_API_KEY``) or ``~/.netrc`` — this script never writes credentials.

Usage::

    WANDB_BASE_URL=https://far.wandb.io uv run python scripts/export_wandb_runs.py \
        --out /tmp/wandb_export
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The eight runs behind docs/status/experiments.md, in campaign order.
RUNS: dict[str, str] = {
    "peract2_base_b200": "7vjpod6m",
    "peract2_base_nhist3_b200": "bk6j5v66",
    "peract2_base_nhist3_clip_b200": "iqa4wuqb",
    "upstream_peract2_repro": "rr1qjj1l",
    "peract2_orbital_nhist3_b200": "pnvpafcg",
    "orbital_miscal_base": "aq54hwdi",  # R1a
    "orbital_miscal_deltaM": "2ks5zjmt",  # R1b
    "orbital_miscal_deltaM_eeaux": "9w9w8xwy",  # R1c
}

DEFAULT_ENTITY = "far-wandb"
DEFAULT_PROJECT = "3dfa"
DEFAULT_BASE_URL = "https://far.wandb.io"


def _json_safe(obj):
    """Coerce wandb config/summary values into something json.dump accepts."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def export_run(api, entity: str, project: str, name: str, run_id: str, out_root: Path) -> dict:
    run = api.run(f"{entity}/{project}/{run_id}")
    out_dir = out_root / f"{name}__{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps(_json_safe(dict(run.config)), indent=2, sort_keys=True))
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(dict(run.summary)), indent=2, sort_keys=True))

    meta = {
        "archive_name": name,
        "wandb_id": run.id,
        "wandb_name": run.name,
        "entity": entity,
        "project": project,
        "state": run.state,
        "created_at": str(run.created_at),
        "tags": list(run.tags or []),
        "url": run.url,
        "runtime_s": run.summary.get("_runtime"),
        "last_step": run.summary.get("_step"),
    }
    (out_dir / "meta.json").write_text(json.dumps(_json_safe(meta), indent=2, sort_keys=True))

    # scan_history streams every logged step (history() downsamples).
    rows = list(run.scan_history())
    n_rows = len(rows)
    columns = sorted({k for row in rows for k in row})

    import csv

    with (out_dir / "history.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    wrote_parquet = False
    try:
        import pandas as pd

        pd.DataFrame(rows, columns=columns).to_parquet(out_dir / "history.parquet", index=False)
        wrote_parquet = True
    except Exception as exc:  # pyarrow/pandas absent — CSV is the durable format
        print(f"  [{name}] parquet skipped: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"  [{name}] {n_rows} history rows, {len(columns)} metrics, parquet={wrote_parquet}")
    return {"name": name, "id": run_id, "rows": n_rows, "metrics": len(columns), "parquet": wrote_parquet}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--base-url", default=os.environ.get("WANDB_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--only", nargs="*", help="subset of archive names to export")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_BASE_URL", args.base_url)

    import wandb

    api = wandb.Api(overrides={"base_url": args.base_url})

    targets = RUNS if not args.only else {k: v for k, v in RUNS.items() if k in args.only}
    if not targets:
        print(f"no runs matched --only {args.only}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"exporting {len(targets)} runs from {args.base_url}/{args.entity}/{args.project}")

    results = [export_run(api, args.entity, args.project, name, rid, args.out) for name, rid in targets.items()]

    index = {
        "entity": args.entity,
        "project": args.project,
        "base_url": args.base_url,
        "runs": results,
    }
    (args.out / "INDEX.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {args.out}/INDEX.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
