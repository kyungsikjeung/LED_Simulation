"""Simulation engine that orchestrates the LED panel and diffusion model."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .diffusion import DiffusionModel
from .panel import LEDPanel


@dataclass
class SimulationResult:
    """Container for simulation output data.

    Attributes
    ----------
    luminance_map:
        2-D array of normalized luminance values on the diffuser plane.
    color_map:
        3-D (H × W × 3) array of normalized RGB values on the diffuser plane.
    uniformity_pct:
        SEMI D35 luminance uniformity (%).
    rms_uniformity_pct:
        RMS luminance uniformity (%).
    panel_width_mm:
        Physical panel width used in the simulation.
    panel_height_mm:
        Physical panel height used in the simulation.
    oad_mm:
        Optical Air Distance used in the simulation.
    resolution_mm:
        Spatial resolution of the output maps.
    """

    luminance_map: np.ndarray
    color_map: np.ndarray
    uniformity_pct: float
    rms_uniformity_pct: float
    panel_width_mm: float
    panel_height_mm: float
    oad_mm: float
    resolution_mm: float
    notes: str = field(default="")

    def summary(self) -> str:
        """Return a human-readable summary of the simulation result."""
        h, w = self.luminance_map.shape
        return (
            f"Panel  : {self.panel_width_mm:.0f} × {self.panel_height_mm:.0f} mm\n"
            f"OAD    : {self.oad_mm:.1f} mm\n"
            f"Map    : {w} × {h} px @ {self.resolution_mm:.1f} mm/px\n"
            f"Lum max: {self.luminance_map.max():.4f}\n"
            f"Lum min: {self.luminance_map.min():.4f}\n"
            f"Lum avg: {self.luminance_map.mean():.4f}\n"
            f"Uniformity (SEMI D35): {self.uniformity_pct:.2f}%\n"
            f"RMS Uniformity       : {self.rms_uniformity_pct:.2f}%"
        )


class Simulation:
    """LED backlight simulation engine.

    Parameters
    ----------
    panel:
        The :class:`~led_simulation.panel.LEDPanel` to simulate.
    oad_mm:
        Optical Air Distance between the LED PCB and the diffuser plate (mm).
    resolution_mm:
        Spatial resolution of the output luminance/color maps (mm/pixel).
    """

    def __init__(
        self,
        panel: LEDPanel,
        oad_mm: float = 30.0,
        resolution_mm: float = 1.0,
    ) -> None:
        self.panel = panel
        self.diffusion = DiffusionModel(oad_mm=oad_mm, resolution_mm=resolution_mm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, notes: str = "") -> SimulationResult:
        """Execute the simulation and return a :class:`SimulationResult`.

        Parameters
        ----------
        notes:
            Optional free-text annotation stored in the result.
        """
        leds = self.panel.all_leds()

        luminance_map = self.diffusion.compute_luminance_map(
            leds,
            self.panel.panel_width_mm,
            self.panel.panel_height_mm,
        )

        color_map = self.diffusion.compute_color_map(
            leds,
            self.panel.panel_width_mm,
            self.panel.panel_height_mm,
        )

        uniformity = DiffusionModel.uniformity(luminance_map)
        rms_uniformity = DiffusionModel.rms_uniformity(luminance_map)

        return SimulationResult(
            luminance_map=luminance_map,
            color_map=color_map,
            uniformity_pct=uniformity,
            rms_uniformity_pct=rms_uniformity,
            panel_width_mm=self.panel.panel_width_mm,
            panel_height_mm=self.panel.panel_height_mm,
            oad_mm=self.diffusion.oad_mm,
            resolution_mm=self.diffusion.resolution_mm,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Convenience class methods
    # ------------------------------------------------------------------

    @classmethod
    def quick_run(
        cls,
        rows: int = 6,
        cols: int = 8,
        panel_width_mm: float = 300.0,
        panel_height_mm: float = 200.0,
        oad_mm: float = 30.0,
        resolution_mm: float = 1.0,
    ) -> SimulationResult:
        """Create a default panel and run a simulation in one call."""
        panel = LEDPanel(
            rows=rows,
            cols=cols,
            panel_width_mm=panel_width_mm,
            panel_height_mm=panel_height_mm,
        )
        sim = cls(panel, oad_mm=oad_mm, resolution_mm=resolution_mm)
        return sim.run()
