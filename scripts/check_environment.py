"""Clean-environment dependency check.

Usage (PowerShell, from the repository root):

    .\\.venv\\Scripts\\python.exe scripts\\check_environment.py
    .\\.venv\\Scripts\\python.exe scripts\\check_environment.py --json --output runs\\environment_check.json
    .\\.venv\\Scripts\\python.exe scripts\\check_environment.py --write-constraints constraints\\verified-py311-windows.txt

Exit code 0 means the checkout resolves the reviewable, reproducible environment.
Exit code 1 means at least one declared dependency is unpinned, missing, or divergent.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from tinybench_lm.environment import (
    CONSTRAINTS_PATH,
    PYPROJECT_PATH,
    check_environment,
    format_report,
    installed_versions,
    render_constraints,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT_PATH)
    parser.add_argument("--constraints", type=Path, default=CONSTRAINTS_PATH)
    parser.add_argument("--json", action="store_true", help="emit the machine-readable report")
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report to this path")
    parser.add_argument(
        "--write-constraints",
        type=Path,
        default=None,
        help="regenerate a constraints file from the versions installed right now",
    )
    parser.add_argument("--no-facts", action="store_true", help="skip informational GPU/backend facts")
    args = parser.parse_args()

    if args.write_constraints is not None:
        args.write_constraints.parent.mkdir(parents=True, exist_ok=True)
        content = render_constraints(
            installed_versions(),
            python_version=platform.python_version(),
            platform_name=platform.platform(),
        )
        args.write_constraints.write_text(content, encoding="utf-8")
        print(f"Wrote observed constraints: {args.write_constraints}")

    report = check_environment(args.pyproject, args.constraints, include_facts=not args.no_facts)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_report(report))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote report: {args.output}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
