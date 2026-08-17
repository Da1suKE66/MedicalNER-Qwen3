#!/usr/bin/env python3
"""Create a deterministic ICD source-record-disjoint train/dev split."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--provenance", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-fraction", type=float, default=0.1)
    ap.add_argument("--suffix", default="_groupdisjoint")
    args = ap.parse_args()

    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    if not all("snapshot_provenance" in row for row in provenance):
        raise ValueError("every provenance row must contain snapshot_provenance")
    if not provenance:
        raise ValueError("provenance is empty")
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(provenance):
        sp = row["snapshot_provenance"]
        key = str(sp.get("source_id") or sp.get("source_code") or "")
        if not key:
            raise ValueError(f"missing source group at index {idx}")
        groups.setdefault(key, []).append(idx)
    keys = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    eval_group_count = max(1, round(len(keys) * args.eval_fraction))
    eval_keys = set(keys[:eval_group_count])
    eval_indices = sorted(i for key in eval_keys for i in groups[key])
    train_indices = sorted(i for key in keys[eval_group_count:] for i in groups[key])
    if set(train_indices) & set(eval_indices):
        raise AssertionError("group split crossed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for input_name in args.input:
        path = Path(input_name)
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) != len(provenance):
            raise ValueError(f"length mismatch: {path} has {len(rows)}, provenance has {len(provenance)}")
        stem = path.stem
        train_path = args.output_dir / f"{stem}{args.suffix}_train.json"
        dev_path = args.output_dir / f"{stem}{args.suffix}_dev.json"
        train_path.write_text(json.dumps([rows[i] for i in train_indices], ensure_ascii=False, indent=2) + "\n")
        dev_path.write_text(json.dumps([rows[i] for i in eval_indices], ensure_ascii=False, indent=2) + "\n")
        outputs.append({
            "input": str(path),
            "input_sha256": digest(path),
            "train": str(train_path),
            "dev": str(dev_path),
            "train_count": len(train_indices),
            "dev_count": len(eval_indices),
            "train_sha256": digest(train_path),
            "dev_sha256": digest(dev_path),
        })
    manifest = {
        "seed": args.seed,
        "eval_fraction_by_source_record": args.eval_fraction,
        "provenance": str(args.provenance),
        "provenance_sha256": digest(args.provenance),
        "records": len(provenance),
        "source_groups": len(groups),
        "train_groups": len(keys) - eval_group_count,
        "dev_groups": eval_group_count,
        "train_indices": train_indices,
        "dev_indices": eval_indices,
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "groupdisjoint_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("records", "source_groups", "train_groups", "dev_groups")}, indent=2))


if __name__ == "__main__":
    main()
