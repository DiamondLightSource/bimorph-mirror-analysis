import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from bimorph_mirror_analysis import __version__
from bimorph_mirror_analysis.__main__ import app, calculate_optimal_voltages
from bimorph_mirror_analysis.read_file import DetectorDimension

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
            baseline_voltage_scan=0,
            slit_range=None,
            detector_dimension=None,
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
            baseline_voltage_scan=0,
            slit_range=None,
            detector_dimension=None,
        )
        assert "The optimal voltages are: [72.14, 50.98, 18.59]" in result.stdout


@pytest.mark.parametrize(
    ["slit_range"],
    [
        ["1.1 8.5"],
        [False],
    ],
)
def test_slit_range_option(slit_range: str | bool, raw_data_pivoted: pd.DataFrame):
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
        mock_read_bluesky_plan_output.return_value = (
            raw_data_pivoted,
            [0, 0, 0],
            100,
            "slits-x_centre",
            "CentroidX",
        )
        mock_calculate_optimal_voltages.side_effect = calculate_optimal_voltages

        if type(slit_range) is str:
            result = runner.invoke(
                app,
                [
                    "calculate-voltages",
                    "tests/data/raw_data.csv",
                    "-1000",
                    "1000",
                    "500",
                    "--slit-range",
                    slit_range.split(" ")[0],
                    slit_range.split(" ")[1],
                ],
            )
            mock_calculate_optimal_voltages.assert_called_with(
                "tests/data/raw_data.csv",
                voltage_range=(-1000, 1000),
                max_consecutive_voltage_difference=500,
                baseline_voltage_scan=0,
                slit_range=(
                    float(slit_range.split(" ")[0]),
                    float(slit_range.split(" ")[1]),
                ),
                detector_dimension=None,
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
            mock_calculate_optimal_voltages.assert_called_with(
                "tests/data/raw_data.csv",
                voltage_range=(-1000, 1000),
                max_consecutive_voltage_difference=500,
                baseline_voltage_scan=0,
                slit_range=None,
                detector_dimension=None,
            )
            assert "The optimal voltages are: [72.14, 50.98, 18.59]" in result.stdout
        mock_np_save.assert_called_once()


@pytest.mark.parametrize(
    ["detector_dimension"],
    [
        ["X"],
        ["Y"],
        ["x"],
        ["y"],
        [None],
    ],
)
def test_detector_dimension_option(
    detector_dimension: str | None, raw_data_pivoted: pd.DataFrame
):
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
        mock_read_bluesky_plan_output.return_value = (
            raw_data_pivoted,
            [0, 0, 0],
            100,
            "slits-x_centre",
            "CentroidX",
        )
        mock_calculate_optimal_voltages.side_effect = calculate_optimal_voltages

        if type(detector_dimension) is str:
            result = runner.invoke(
                app,
                [
                    "calculate-voltages",
                    "tests/data/raw_data.csv",
                    "-1000",
                    "1000",
                    "500",
                    "--detector-dimension",
                    detector_dimension,
                ],
            )
            mock_calculate_optimal_voltages.assert_called_with(
                "tests/data/raw_data.csv",
                voltage_range=(-1000, 1000),
                max_consecutive_voltage_difference=500,
                baseline_voltage_scan=0,
                slit_range=None,
                detector_dimension=DetectorDimension(detector_dimension.upper()),
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
            mock_calculate_optimal_voltages.assert_called_with(
                "tests/data/raw_data.csv",
                voltage_range=(-1000, 1000),
                max_consecutive_voltage_difference=500,
                baseline_voltage_scan=0,
                slit_range=None,
                detector_dimension=None,
            )
            assert "The optimal voltages are: [72.14, 50.98, 18.59]" in result.stdout
        mock_np_save.assert_called_once()


@pytest.mark.parametrize(
    ["output_dir", "detector_dimension"],
    [
        ["outdir", "X"],
        ["outdir/", "X"],
        ["outdir/", "Y"],
        ["outdir/", "x"],
        ["outdir/", "y"],
        ["outdir/", None],
    ],
)
def test_generate_plots(
    raw_data_pivoted: pd.DataFrame, output_dir: str, detector_dimension: str | None
):
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
        if detector_dimension is not None:
            mock_read_bluesky_plan_output.return_value = (
                raw_data_pivoted,
                [0, 0, 0],
                100,
                "slits-x_centre",
                f"Centroid{detector_dimension.upper()}",
            )

            result = runner.invoke(
                app,
                [
                    "generate-plots",
                    "input.csv",
                    output_dir,
                    "-1000",
                    "1000",
                    "500",
                    "--baseline-voltage-scan",
                    "0",
                    "--detector-dimension",
                    detector_dimension,
                ],
            )

            assert result.exit_code == 0

            mock_read_bluesky_plan_output.assert_called_once_with(
                "input.csv",
                baseline_voltage_scan_index=0,
                detector_dimension=DetectorDimension(detector_dimension.upper()),
            )

        else:
            mock_read_bluesky_plan_output.return_value = (
                raw_data_pivoted,
                [0, 0, 0],
                100,
                "slits-x_centre",
                "CentroidX",
            )
            result = runner.invoke(
                app,
                [
                    "generate-plots",
                    "input.csv",
                    output_dir,
                    "-1000",
                    "1000",
                    "500",
                    "--baseline-voltage-scan",
                    "0",
                ],
            )

            assert result.exit_code == 0

            mock_read_bluesky_plan_output.assert_called_once_with(
                "input.csv", baseline_voltage_scan_index=0, detector_dimension=None
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
