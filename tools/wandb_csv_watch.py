"""Stream BlinkMimic's CSV metrics to one W&B run without changing training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="training CSV written by blinkmimic.train")
    parser.add_argument("--done-file", type=Path, required=True, help="created by the training wrapper after exit")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--config", type=Path, help="optional args.json written by the trainer")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open() as stream:
        value = json.load(stream)
    return value if isinstance(value, dict) else {}


def read_rows(path: Path, offset: int) -> tuple[list[dict[str, object]], int]:
    if not path.exists():
        return [], offset
    with path.open(newline="") as stream:
        stream.seek(offset)
        text = stream.read()
    if not text:
        return [], offset

    last_newline = text.rfind("\n")
    if last_newline < 0:
        return [], offset
    complete = text[: last_newline + 1]
    new_offset = offset + len(complete)
    lines = complete.splitlines()
    if offset == 0:
        reader = csv.DictReader(lines)
        rows = list(reader)
    else:
        # The header was consumed on the first pass.  Read it again so an
        # append that ends with a partial line is handled on the next pass.
        with path.open(newline="") as full_stream:
            header = next(csv.reader(full_stream), [])
        reader = csv.DictReader(lines, fieldnames=header)
        rows = list(reader)
    return rows, new_offset


def numeric_row(row: dict[str, object]) -> dict[str, object] | None:
    result: dict[str, object] = {}
    for key, value in row.items():
        if value in (None, ""):
            continue
        if key == "iteration":
            try:
                result[key] = int(str(value))
            except ValueError:
                return None
            continue
        try:
            number = float(str(value))
        except ValueError:
            continue
        if math.isfinite(number):
            result[key] = number
    return result if "iteration" in result else None


def main() -> None:
    args = parse_args()
    if not args.project:
        raise SystemExit("--project or WANDB_PROJECT is required")

    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        mode=args.mode,
        config=load_config(args.config),
    )
    run.define_metric("iteration")
    offset = 0
    while True:
        rows, offset = read_rows(args.csv, offset)
        for row in rows:
            metrics = numeric_row(row)
            if metrics is None:
                continue
            run.log(metrics, step=int(metrics["iteration"]))
        if args.done_file.exists() and not rows:
            break
        time.sleep(max(args.poll_seconds, 0.1))
    run.finish()


if __name__ == "__main__":
    main()
