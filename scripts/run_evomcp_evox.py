#!/usr/bin/env python3
"""Run evomcp EvoX with the project-specific ecdsa.fail evaluator."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--evomcp-root", type=Path, default=Path("/private/tmp/evomcp"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.evomcp_root))
    sys.path.insert(0, str(project_root))

    import optim.search_spaces  # noqa: F401
    from evomcp.optim.evox_runner import run
    from optim.evaluators.ecdsafail import EcdsaFailEvaluator

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    archive = run(args.config, evaluator=EcdsaFailEvaluator(project_root), resume=args.resume)
    best = archive.best()
    if best:
        candidate, result = best
        print("BEST", candidate.candidate_id, result.primary_score, result.secondary_scores)


if __name__ == "__main__":
    main()
