"""Reconcile the frozen source registry against the authoritative corpus contract.

    python scripts/audit_source_policy.py
    python scripts/audit_source_policy.py --json

Exit code 0 means the frozen registry still matches Plan Sections 4.1-4.5, 5.2, and 11.2:
stable shares and reserved margins reconcile, the manifest schema preserves every required
provenance field, and every named prohibition is encoded. Operator-gated prerequisites are
reported as BLOCKED or DEFERRED with their blocker, owner, and next action; they are not
failures and they are never silently promoted to a pass.

This script reads local configs only. It does not acquire, download, or approve a corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinybench_lm.data_protocols import protocol_digest
from tinybench_lm.source_manifest import (
    FILTERS_PROTOCOL_PATH,
    SOURCES_PROTOCOL_PATH,
    audit_source_policy,
    format_source_policy_report,
    load_filter_protocol,
    load_source_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", type=Path, default=SOURCES_PROTOCOL_PATH)
    parser.add_argument("--filters", type=Path, default=FILTERS_PROTOCOL_PATH)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable results")
    args = parser.parse_args()

    registry = load_source_registry(args.sources)
    load_filter_protocol(args.filters)
    results = audit_source_policy(registry)
    failures = [result for result in results if result.failed]

    if args.json:
        payload = {
            "ok": not failures,
            "sources_digest": protocol_digest(args.sources),
            "filters_digest": protocol_digest(args.filters),
            "results": [result.__dict__ for result in results],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"sources digest: {protocol_digest(args.sources)}")
        print(f"filters digest: {protocol_digest(args.filters)}")
        print(format_source_policy_report(results))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
