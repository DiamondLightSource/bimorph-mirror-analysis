import subprocess
import sys

from bimorph_mirror_analysis import __version__


def test_cli_version():
    cmd = [
        sys.executable,
        "-m",
        "bimorph_mirror_analysis",
        "tests/data/raw_data.csv",
        "--version",
    ]
    assert subprocess.check_output(cmd).decode().strip() == __version__
