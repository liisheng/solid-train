"""Local command line for durable-checkpoint verification and retention planning.

Everything here is local, bounded, and non-destructive (Plan Sections 7.2, 9, 13 G2):

    python scripts/verify_checkpoint.py verify runs/<run>/latest.pt
    python scripts/verify_checkpoint.py verify runs/<run>/latest.pt --expected-run-id run-abc123
    python scripts/verify_checkpoint.py inspect runs/<run>/latest.pt
    python scripts/verify_checkpoint.py retention runs/<run>
    python scripts/verify_checkpoint.py readiness

`retention` prints a proposal only. This script never deletes a local file and never touches
a remote copy: off-machine copy verification stays an operator action and is reported as
NOT_RUN rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinybench_lm.checkpointing import (
    ROLE_BRANCH_PARENT,
    ROLE_FALLBACK,
    ROLE_SELECTED_ENDPOINT,
    format_checkpoint_report,
    format_retention_plan,
    inventory_from_directory,
    plan_retention,
    read_manifest,
    readiness_results,
    retention_plan_violations,
    verify_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="Emit machine-readable results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="Re-derive every claim a checkpoint makes")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expected-run-id")
    verify.add_argument("--expected-schedule-hash")

    inspect = subparsers.add_parser("inspect", help="Print a checkpoint's sidecar manifest")
    inspect.add_argument("path", type=Path)

    retention = subparsers.add_parser(
        "retention", help="Propose which local copies to keep (never deletes anything)"
    )
    retention.add_argument("directory", type=Path)
    retention.add_argument("--branch-parent", action="append", default=[], help="filename to protect")
    retention.add_argument("--selected-endpoint", action="append", default=[])
    retention.add_argument("--fallback", action="append", default=[])
    retention.add_argument("--keep-latest", type=int)

    subparsers.add_parser("readiness", help="Report unmeasured checkpoint evidence as NOT_RUN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "inspect":
        manifest = read_manifest(args.path)
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        return

    if args.command == "readiness":
        results = readiness_results()
        if args.json:
            print(json.dumps([result.__dict__ for result in results], indent=2, sort_keys=True))
        else:
            print(format_checkpoint_report(results, "Checkpoint evidence readiness:"))
        return

    if args.command == "retention":
        roles: dict[str, str] = {}
        for name in args.branch_parent:
            roles[name] = ROLE_BRANCH_PARENT
        for name in args.selected_endpoint:
            roles[name] = ROLE_SELECTED_ENDPOINT
        for name in args.fallback:
            roles[name] = ROLE_FALLBACK
        entries = inventory_from_directory(args.directory, roles=roles)
        plan = plan_retention(entries, keep_latest=args.keep_latest)
        problems = retention_plan_violations(plan, entries)
        if args.json:
            print(
                json.dumps(
                    {
                        "inventory": [entry.to_dict() for entry in entries],
                        "plan": plan.to_dict(),
                        "violations": list(problems),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(format_retention_plan(plan))
            for problem in problems:
                print(f"  UNSAFE: {problem}")
        raise SystemExit(1 if problems else 0)

    report = verify_checkpoint(
        args.path,
        expected_run_id=args.expected_run_id,
        expected_schedule_content_hash=args.expected_schedule_hash,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_checkpoint_report(list(report.results), f"Checkpoint verification: {args.path}"))
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
