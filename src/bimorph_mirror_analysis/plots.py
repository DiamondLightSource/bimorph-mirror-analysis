import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Plot:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))  # type: ignore
        self.ax = self.fig.add_subplot(111)  # type: ignore
        self.ax.spines["top"].set_visible(False)  # type: ignore
        self.ax.spines["right"].set_visible(False)  # type: ignore

    def save_plot(self, filename: str):
        self.fig.savefig(filename)  # type: ignore


class InfluenceFunctionPlot(Plot):
    def __init__(
        self,
        slit_positions: np.typing.NDArray[np.float64],
        centroids: np.typing.NDArray[np.float64],
        actuator_num: int,
    ):
        super().__init__()
        self.ax.set_xlabel("Slit position", fontsize=18)  # type: ignore
        self.ax.set_ylabel("Affect on Centroid Position", fontsize=18)  # type: ignore
        self.ax.set_title(  # type: ignore
            f"Influence Function of Actuator {actuator_num}", fontsize=24, pad=30
        )
        self.ax.plot(  # type: ignore
            [min(slit_positions), max(slit_positions)],
            [0, 0],
            color="black",
            linestyle=":",
            alpha=0.6,
        )
        self.ax.plot(slit_positions, centroids)  # type: ignore


class MirrorSurfacePlot(Plot):
    def __init__(
        self,
        slit_positions: np.typing.NDArray[np.float64],
        baseline_centroids: np.typing.NDArray[np.float64],
        unrestrained_predicted_centroids: np.typing.NDArray[np.float64] | None = None,
        restrained_predicted_centroids: np.typing.NDArray[np.float64] | None = None,
    ):
        super().__init__()
        self.ax.set_xlabel("Slit position", fontsize=18)  # type: ignore
        self.ax.set_ylabel("Centroid position", fontsize=18)  # type: ignore
        self.ax.set_title("Mirror Surface Plot", fontsize=24, pad=30)  # type: ignore
        self.ax.plot(slit_positions, baseline_centroids, label="Baseline")  # type: ignore
        if unrestrained_predicted_centroids is not None:
            self.ax.plot(  # type: ignore
                slit_positions,
                unrestrained_predicted_centroids,
                label="Predicted, unrestrained",
            )
        if restrained_predicted_centroids is not None:
            self.ax.plot(  # type: ignore
                slit_positions,
                restrained_predicted_centroids,
                label="Predicted, restrained",
            )
        self.ax.legend()  # type: ignore


class PencilBeamScanPlot(Plot):
    def __init__(self, pivoted_df: pd.DataFrame, scan_num: int):
        super().__init__()
        self.ax.set_xlabel("Slit position", fontsize=18)  # type: ignore
        self.ax.set_ylabel("Centroid position", fontsize=18)  # type: ignore
        self.ax.set_title(f"Beamline Scan {scan_num}", fontsize=24, pad=30)  # type: ignore
        self.ax.plot(pivoted_df[f"pencil_beam_scan_{scan_num}"])  # type: ignore
