from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bimorph_mirror_analysis.plots import (
    InfluenceFunctionPlot,
    MirrorSurfacePlot,
    PencilBeamScanPlot,
    Plot,
)


@pytest.fixture
def sample_slit_positions() -> np.typing.NDArray[np.float64]:
    # just any data to plot works
    return np.linspace(0, 10, 100, dtype=np.float64)  # type: ignore


@pytest.fixture
def sample_centroids() -> np.typing.NDArray[np.float64]:
    # just any data to plot works
    return np.sin(np.linspace(0, 10, 100))  # type: ignore


def test_influence_function_plot(
    sample_slit_positions: np.typing.NDArray[np.float64],
    sample_centroids: np.typing.NDArray[np.float64],
):
    actuator_num = 1
    plot = InfluenceFunctionPlot(sample_slit_positions, sample_centroids, actuator_num)

    # Verify the title and labels
    assert plot.ax.get_xlabel() == "Slit position"
    assert plot.ax.get_ylabel() == "Affect on Centroid Position"
    assert plot.ax.get_title() == f"Influence Function of Actuator {actuator_num}"

    # Verify the data plotted
    lines = plot.ax.get_lines()
    assert len(lines) == 2  # One for the zero line, one for the centroids
    assert np.array_equal(
        lines[0].get_xdata(), [min(sample_slit_positions), max(sample_slit_positions)]
    )
    assert np.array_equal(lines[0].get_ydata(), [0, 0])
    assert np.array_equal(lines[1].get_xdata(), sample_slit_positions)
    assert np.array_equal(lines[1].get_ydata(), sample_centroids)


def test_mirror_surface_plot(
    sample_slit_positions: np.typing.NDArray[np.float64],
    sample_centroids: np.typing.NDArray[np.float64],
):
    # just change the data for predicted centroids slightly.
    unrestrained_predicted_centroids = sample_centroids + 0.1
    restrained_predicted_centroids = sample_centroids - 0.1
    plot = MirrorSurfacePlot(
        sample_slit_positions,
        sample_centroids,
        unrestrained_predicted_centroids,
        restrained_predicted_centroids,
    )

    # Verify the title and labels
    assert plot.ax.get_xlabel() == "Slit position"
    assert plot.ax.get_ylabel() == "Centroid position"
    assert plot.ax.get_title() == "Mirror Surface Plot"

    # Verify the data plotted
    lines = plot.ax.get_lines()
    assert len(lines) == 3  # Baseline, unrestrained, and restrained
    assert np.array_equal(lines[0].get_xdata(), sample_slit_positions)
    assert np.array_equal(lines[0].get_ydata(), sample_centroids)

    assert np.array_equal(lines[1].get_xdata(), sample_slit_positions)
    assert np.array_equal(lines[1].get_ydata(), unrestrained_predicted_centroids)

    assert np.array_equal(lines[2].get_xdata(), sample_slit_positions)
    assert np.array_equal(lines[2].get_ydata(), restrained_predicted_centroids)

    # Verify the legend
    legend_labels = [text.get_text() for text in plot.ax.get_legend().get_texts()]
    assert legend_labels == [
        "Baseline",
        "Predicted, unrestrained",
        "Predicted, restrained",
    ]


def test_pencil_beam_scan_plot(raw_data_pivoted: pd.DataFrame):
    scan_num = 1
    plot = PencilBeamScanPlot(raw_data_pivoted, scan_num)  # type: ignore

    # Verify the title and labels
    assert plot.ax.get_xlabel() == "Slit position"
    assert plot.ax.get_ylabel() == "Centroid position"
    assert plot.ax.get_title() == f"Beamline Scan {scan_num}"

    # Verify the data plotted
    lines = plot.ax.get_lines()
    assert len(lines) == 1
    assert np.array_equal(
        lines[0].get_xdata(),
        raw_data_pivoted["slits-x_centre"].to_numpy(),  # type: ignore
    )
    assert np.array_equal(
        lines[0].get_ydata(),
        raw_data_pivoted[f"pencil_beam_scan_{scan_num}"],  # type: ignore
    )


def test_save_plot():
    plot = Plot()
    with (
        patch.object(plot.fig, "savefig") as mock_savefig,
    ):
        plot.save_plot("output_directory/filename")
        mock_savefig.assert_called_once()
