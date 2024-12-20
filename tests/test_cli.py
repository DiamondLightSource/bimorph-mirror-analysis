import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from bimorph_mirror_analysis import __version__
from bimorph_mirror_analysis.__main__ import app

runner = CliRunner()


@pytest.mark.parametrize(
    ["outpath"],
    [
        ["tests/data/out.csv"],
        [False],
    ],
)
def test_app(outpath: str | bool):
    with (
        patch("bimorph_mirror_analysis.__main__.np.savetxt") as mock_np_save,
        patch(
            "bimorph_mirror_analysis.__main__.calculate_optimal_voltages"
        ) as mock_calculate_optimal_voltages,
    ):
        mock_calculate_optimal_voltages.return_value = np.array([72.14, 50.98, 18.59])
        if outpath is not False:
            print("\n\n\n\n\n\n")
            result = runner.invoke(
                app,
                [
                    "calculate-voltages",
                    "tests/data/raw_data.csv",
                    "--output_path",
                    f"{outpath}",
                ],
            )
            print("aaa")
        elif not outpath:
            result = runner.invoke(
                app, ["calculate-voltages", "tests/data/raw_data.csv"]
            )
        mock_np_save.assert_called_once()
        mock_calculate_optimal_voltages.assert_called_with("tests/data/raw_data.csv")
        assert "The optimal voltages are: [72.14, 50.98, 18.59]" in result.stdout


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
