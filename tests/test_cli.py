from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from massgenqr import cli


@pytest.fixture(autouse=True)
def stub_dependencies(monkeypatch):
    """Avoid importing heavy optional dependencies during tests."""

    monkeypatch.setattr(cli, "ensure_dependencies", lambda: None)


def test_main_parser_error_empty_alphabet(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["1", "--alphabet", ""])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "error: alphabet must not be empty" in captured.err


def test_main_parser_error_count_exceeds_capacity(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["3", "--alphabet", "A", "--id-length", "1"])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "Maximum distinct identifiers: 1" in captured.err
