from __future__ import annotations

from pathlib import Path

from succession_fragility.utils.secrets import scan_file


def test_secret_scan_flags_obvious_key(tmp_path: Path) -> None:
    target = tmp_path / "bad.txt"
    fake_value = "abc" + "def" + "ghi" + "jkl" + "mno" + "pqr"
    target.write_text(f"FRED_API_KEY={fake_value}\n", encoding="utf-8")
    assert scan_file(target)


def test_secret_scan_ignores_placeholders(tmp_path: Path) -> None:
    target = tmp_path / "ok.txt"
    target.write_text("FRED_API_KEY=\nSEC_USER_AGENT='Name email@example.com'\n", encoding="utf-8")
    assert scan_file(target) == []
