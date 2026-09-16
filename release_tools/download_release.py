"""Download and verify the published baseline, using only Python's standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/release"))
    args = parser.parse_args()
    manifest_path = Path(__file__).resolve().parents[1] / "configs/release/submission_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        destination = args.output_dir / item["name"]
        if destination.exists() and sha256(destination) == item["sha256"]:
            print(f"Verified existing {destination.name}")
            continue
        temporary = destination.with_suffix(destination.suffix + ".download")
        request = urllib.request.Request(item["url"], headers={"User-Agent": "TinyBench-LM-release"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
        if temporary.stat().st_size != item["bytes"] or sha256(temporary) != item["sha256"]:
            temporary.unlink()
            raise SystemExit(f"Integrity check failed: {destination.name}")
        temporary.replace(destination)
        print(f"Downloaded and verified {destination.name}")


if __name__ == "__main__":
    main()
