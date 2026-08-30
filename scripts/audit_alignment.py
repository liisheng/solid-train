"""Report whether the repository matches the authoritative plan.

Read-only by construction: this prints a report and exits non-zero on an unexplained
difference. It never writes, formats, or repairs anything, so running it twice is safe and
returns the identical result.

    python scripts/audit_alignment.py
    python scripts/audit_alignment.py --check-idempotence

Exit codes:
    0  no unexplained difference (deferrals are explained, not failures)
    1  at least one unexplained difference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tinybench_lm.alignment import (  # noqa: E402
    audit_is_idempotent,
    audit_repository,
    format_audit_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit the report as JSON instead of a status block"
    )
    parser.add_argument(
        "--check-idempotence",
        action="store_true",
        help="audit twice and prove the report and the repository are both unchanged",
    )
    arguments = parser.parse_args()

    report = audit_repository(root=REPOSITORY_ROOT)
    if arguments.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_audit_report(report))

    exit_code = 0 if report.ok else 1

    if arguments.check_idempotence:
        idempotent, problems = audit_is_idempotent(root=REPOSITORY_ROOT)
        print()
        if idempotent:
            print("IDEMPOTENT  a second audit returned the same report and changed no files")
        else:
            for problem in problems:
                print(f"NOT IDEMPOTENT  {problem}")
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
