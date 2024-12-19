import subprocess
import sys
from unittest.mock import patch

from typer.testing import CliRunner

from bimorph_mirror_analysis import __version__
from bimorph_mirror_analysis.__main__ import app

runner = CliRunner()


def test_app():
    with patch("bimorph_mirror_analysis.__main__.np.savetxt") as mock_np_save:
        runner.invoke(app, ["calculate-voltages", "tests/data/raw_data.csv"])
        mock_np_save.assert_called_once()
        runner.invoke(app, ["calculate-voltages", "tests/data/raw_data.csv"])


def test_cli_version():
    cmd = [
        sys.executable,
        "-m",
        "bimorph_mirror_analysis",
        "--version",
    ]
    assert (
        subprocess.check_output(cmd).decode().strip("Version: ").strip("\n")
        == __version__
    )
