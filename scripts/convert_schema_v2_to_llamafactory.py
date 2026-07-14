#!/usr/bin/env python3
"""CLI wrapper for the strict Schema v2 LLaMAFactory training gate."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.convert_schema_v2_to_llamafactory import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
