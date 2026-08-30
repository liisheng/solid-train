"""Inspect, verify, and promote frozen evaluation protocols (Plan Sections 2.2, 10.1-10.4).

Three bounded actions, no network and no benchmark execution:

    --verify                 run every frozen required check on a protocol
    --hash                   print the protocol identity hash a score must cite
    --promote answers.json   create a NEW official protocol from organizer answers
    --verify-bundle DIR      verify one written evidence bundle

Promotion never edits the provisional protocol. It writes
``configs/evaluation/evaluation_organizer_final_v1.yaml`` plus a digest sidecar, refuses to
overwrite an existing promoted protocol, and re-checks the provisional digest afterwards.

Answer file shape (every key is required; a partial answer set is refused)::

    {
      "answered_on": "2026-02-01",
      "source": "organizer email thread",
      "num_fewshot": {"hellaswag": 0, "arc_easy": 0, "piqa": 0, "winogrande": 0,
                      "wikitext_103_perplexity": 0},
      "metric_keys": {"hellaswag": ["acc_norm"], "arc_easy": ["acc"], "piqa": ["acc"],
                      "winogrande": ["acc"], "wikitext_103_perplexity": ["word_perplexity"]},
      "wikitext_103": {"split": "test", "slice": "full_split",
                       "normalization": "organizer_specified", "bos_eos_handling": "...",
                       "context_length": 1024, "stride": 512, "denominator": "..."},
      "judges_rerun_policy": "...",
      "own_weight_upload_policy": "...",
      "harness_commit": "",           # optional operator pin; empty stays BLOCKED
      "dataset_revisions": {}         # optional operator pins; empty stays BLOCKED
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tinybench_lm.evaluation_protocol import (  # noqa: E402
    PROVISIONAL_PROTOCOL_PATH,
    EvaluationProtocolError,
    OrganizerAnswers,
    format_report,
    load_evaluation_protocol,
    promote_to_organizer_final,
    protocol_identity,
    verify_evaluation_protocol,
    verify_run_bundle,
)


def _load_answers(path: Path) -> OrganizerAnswers:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return OrganizerAnswers(
        num_fewshot={str(key): int(value) for key, value in payload.get("num_fewshot", {}).items()},
        metric_keys={str(key): list(value) for key, value in payload.get("metric_keys", {}).items()},
        wikitext_103=dict(payload.get("wikitext_103", {})),
        judges_rerun_policy=str(payload.get("judges_rerun_policy", "")),
        own_weight_upload_policy=str(payload.get("own_weight_upload_policy", "")),
        answered_on=str(payload.get("answered_on", "")),
        source=str(payload.get("source", "")),
        harness_commit=str(payload.get("harness_commit", "")),
        dataset_revisions={str(key): str(value) for key, value in payload.get("dataset_revisions", {}).items()},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect, verify, and promote evaluation protocols")
    parser.add_argument("--protocol", type=Path, default=PROVISIONAL_PROTOCOL_PATH)
    parser.add_argument("--verify", action="store_true", help="Run every frozen required check")
    parser.add_argument("--hash", action="store_true", help="Print the protocol identity")
    parser.add_argument("--promote", type=Path, help="Organizer answer JSON file")
    parser.add_argument("--output", type=Path, help="Promotion output path (defaults to the protocol's target_path)")
    parser.add_argument("--verify-bundle", type=Path, help="Verify a written evidence bundle directory")
    args = parser.parse_args()

    if not (args.verify or args.hash or args.promote or args.verify_bundle):
        args.verify = True

    protocol = load_evaluation_protocol(args.protocol)
    status = 0

    if args.hash:
        print(json.dumps(protocol_identity(protocol).to_dict(), indent=2, sort_keys=True))

    if args.verify:
        report = verify_evaluation_protocol(protocol, path=args.protocol)
        print(format_report(report))
        status = max(status, 0 if report.ok else 1)

    if args.promote:
        try:
            result = promote_to_organizer_final(
                _load_answers(args.promote),
                protocol=protocol,
                provisional_path=args.protocol,
                output_path=args.output,
            )
        except EvaluationProtocolError as error:
            print(f"PROMOTION REFUSED: {error}")
            return 1
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        if result.still_blocked:
            print(
                "\nStill BLOCKED after promotion (operator pins, not organizer answers): "
                f"{list(result.still_blocked)}"
            )

    if args.verify_bundle:
        report = verify_run_bundle(args.verify_bundle, protocol)
        print(format_report(report))
        status = max(status, 0 if report.ok else 1)

    return status


if __name__ == "__main__":
    raise SystemExit(main())
