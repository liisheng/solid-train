"""The release downloader must reject damaged weights before installation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "release_tools/download_release.py"


def prepare(tmp_path, monkeypatch, *, valid=True):
    spec = importlib.util.spec_from_file_location("download_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source.pt"
    source.write_bytes(b"verified fixture weights")
    manifest = tmp_path / "configs/release/submission_v1.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"files": [{
        "name": "baseline_export.pt", "url": source.as_uri(),
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest() if valid else "0" * 64,
    }]}))
    monkeypatch.setattr(module, "__file__", str(tmp_path / "release_tools/download_release.py"))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["download_release", "--output-dir", str(output)])
    return module, source, output


def test_download_and_verified_cache(tmp_path, monkeypatch):
    module, source, output = prepare(tmp_path, monkeypatch)
    module.main()
    assert (output / "baseline_export.pt").read_bytes() == source.read_bytes()
    source.unlink()  # A verified cache must succeed without network/source access.
    module.main()


def test_bad_download_preserves_existing_file(tmp_path, monkeypatch):
    module, _, output = prepare(tmp_path, monkeypatch, valid=False)
    output.mkdir()
    target = output / "baseline_export.pt"
    target.write_bytes(b"existing file")
    with pytest.raises(SystemExit, match="Integrity check failed"):
        module.main()
    assert target.read_bytes() == b"existing file"
    assert not target.with_suffix(".pt.download").exists()


def test_distribution_tool_is_not_a_training_dependency():
    from tinybench_lm.eligibility import audit_eligibility, production_python_paths

    root = SCRIPT.parents[1]
    assert SCRIPT not in production_python_paths(root)
    assert audit_eligibility(root).ok
    for path in production_python_paths(root):
        source = path.read_text(encoding="utf-8")
        assert "release_tools" not in source
        assert "download_release" not in source
