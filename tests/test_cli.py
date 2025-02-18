import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
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
def test_outpath_option(outpath: str | bool):
    with (
        patch("bimorph_mirror_analysis.__main__.np.savetxt") as mock_np_save,
        patch(
            "bimorph_mirror_analysis.__main__.calculate_optimal_voltages"
        ) as mock_calculate_optimal_voltages,
    ):
        mock_calculate_optimal_voltages.return_value = np.array([72.14, 50.98, 18.59])
        if type(outpath) is str:
            result = runner.invoke(
                app,
                [
                    "calculate-voltages",
                    "tests/data/raw_data.csv",
                    "-1000",
                    "1000",
                    "500",
                    "--output-path",
                    f"{outpath}",
                ],
            )
        else:
            result = runner.invoke(
                app,
                [
                    "calculate-voltages",
                    "tests/data/raw_data.csv",
                    "-1000",
                    "1000",
                    "500",
                ],
            )
        mock_np_save.assert_called_once()
        mock_calculate_optimal_voltages.assert_called_with(
            "tests/data/raw_data.csv",
            voltage_range=(-1000, 1000),
            max_consecutive_voltage_difference=500,
        )
        assert "The optimal voltages are: [72.14, 50.98, 18.59]" in result.stdout


@pytest.mark.parametrize(
    ["human_readable"],
    [
        ["tests/data/out.csv"],
        [False],
    ],
)
def test_human_readable_option(human_readable: str | bool):
    with (
        patch("bimorph_mirror_analysis.__main__.np.savetxt") as mock_np_save,
        patch(
            "bimorph_mirror_analysis.__main__.calculate_optimal_voltages"
        ) as mock_calculate_optimal_voltages,
        patch(
            "bimorph_mirror_analysis.__main__.read_bluesky_plan_output"
        ) as mock_read_bluesky_plan_output,
    ):
        # Create a mock DataFrame
        mock_pivoted = MagicMock(spec=pd.DataFrame)
        mock_read_bluesky_plan_output.return_value = (mock_pivoted,)
        mock_calculate_optimal_voltages.return_value = np.array([72.14, 50.98, 18.59])

        if type(human_readable) is str:
            result = runner.invoke(
                app,
                [
                    "calculate-voltages",
                    "tests/data/raw_data.csv",
                    "-1000",
                    "1000",
                    "500",
                    "--human-readable",
                    f"{human_readable}",
                ],
            )
            mock_pivoted.to_csv.assert_called_once()
        else:
            result = runner.invoke(
                app,
                [
                    "calculate-voltages",
                    "tests/data/raw_data.csv",
                    "-1000",
                    "1000",
                    "500",
                ],
            )
        mock_np_save.assert_called_once()
        mock_calculate_optimal_voltages.assert_called_with(
            "tests/data/raw_data.csv",
            voltage_range=(-1000, 1000),
            max_consecutive_voltage_difference=500,
        )
        assert "The optimal voltages are: [72.14, 50.98, 18.59]" in result.stdout


@pytest.mark.parametrize("output_dir", ["outdir", "outdir/"])
def test_generate_plots(raw_data_pivoted: pd.DataFrame, output_dir: str):
    with (
        patch(
            "bimorph_mirror_analysis.__main__.InfluenceFunctionPlot.save_plot"
        ) as mock_InfluenceFunctionPlot_save_plot,
        patch(
            "bimorph_mirror_analysis.__main__.MirrorSurfacePlot.save_plot"
        ) as mock_MirrorSurfacePlot_save_plot,
        patch(
            "bimorph_mirror_analysis.__main__.PencilBeamScanPlot.save_plot"
        ) as mock_PencilBeamScanPlot_save_plot,
        patch(
            "bimorph_mirror_analysis.__main__.read_bluesky_plan_output"
        ) as mock_read_bluesky_plan_output,
    ):
        mock_read_bluesky_plan_output.return_value = [raw_data_pivoted, 0, 100]
        _ = runner.invoke(
            app,
            [
                "generate-plots",
                "input.csv",
                output_dir,
                "--baseline-voltage-scan",
                "0",
            ],
        )
        # assert that the slash is added if it is missing
        if output_dir[-1] != "/":
            mock_MirrorSurfacePlot_save_plot.assert_called_once_with(
                f"{output_dir}/mirror_surface_plot.png"
            )
        # assert that slash not added when present
        else:
            mock_MirrorSurfacePlot_save_plot.assert_called_once_with(
                f"{output_dir}mirror_surface_plot.png"
            )

        mock_read_bluesky_plan_output.assert_called_once()
        assert mock_PencilBeamScanPlot_save_plot.call_count == 4
        assert mock_InfluenceFunctionPlot_save_plot.call_count == 3
        mock_MirrorSurfacePlot_save_plot.assert_called_once()


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
