"""Fail-closed eligibility audit of the production model-weight path.

    python scripts/audit_eligibility.py
    python scripts/audit_eligibility.py --json

Exit code 0 means every scanned production source is eligible: random initialization only,
no remote weight retrieval, no knowledge transfer from another model, and no hosted-model
inference dependency (Plan Sections 2, 3.3, 4.5, 13 G0/G6).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinybench_lm.eligibility import audit_eligibility, format_eligibility_report
from tinybench_lm.environment import REPOSITORY_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable results")
    args = parser.parse_args()

    report = audit_eligibility(args.repository_root)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_eligibility_report(report))
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
