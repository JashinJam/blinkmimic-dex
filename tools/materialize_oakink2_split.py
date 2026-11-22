"""Materialize the train/holdout split declared by an OakInk2 Sharpa manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def materialize(data_dir: Path, split: str) -> list[str]:
    manifest_path = data_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    clips = list(manifest.get(split, []))
    if not clips:
        raise ValueError(f"{manifest_path}: split {split!r} is empty")
    target = data_dir / split
    target.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        source = data_dir / f"{clip}.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = target / source.name
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    split_manifest = {
        "format_version": manifest.get("format_version", 1),
        "parent_manifest": str(manifest_path),
        "robot": manifest.get("robot"),
        "object_id": manifest.get("object_id"),
        "split": split,
        "clips": sorted(clips),
    }
    (target / "dataset_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n")
    return sorted(clips)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "holdout", "both"), default="both")
    args = parser.parse_args()
    splits = ("train", "holdout") if args.split == "both" else (args.split,)
    rows = {split: materialize(args.data_dir, split) for split in splits}
    print(json.dumps({"data_dir": str(args.data_dir), "splits": rows}, indent=2))


if __name__ == "__main__":
    main()
