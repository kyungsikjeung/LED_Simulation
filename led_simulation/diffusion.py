"""Light diffusion model used by the simulation engine.

The model approximates the luminance contribution of each LED on the
diffuser plate using a 2-D Gaussian kernel.  The width of the Gaussian
is derived from the LED half-angle and the optical distance (OAD) between
the LED array and the diffuser plate.
"""

import numpy as np


class DiffusionModel:
    """Gaussian light-diffusion model.

    Parameters
    ----------
    oad_mm:
        Optical Air Distance — vertical distance between the LED PCB and
        the diffuser plate in millimetres.  Larger values produce wider,
        more uniform light distributions.
    resolution_mm:
        Spatial resolution of the output luminance map in mm/pixel.
    """

    def __init__(self, oad_mm: float = 30.0, resolution_mm: float = 1.0) -> None:
        if oad_mm <= 0:
            raise ValueError(f"oad_mm must be positive, got {oad_mm}")
        if resolution_mm <= 0:
            raise ValueError(f"resolution_mm must be positive, got {resolution_mm}")

        self.oad_mm = oad_mm
        self.resolution_mm = resolution_mm

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_luminance_map(
        self,
        leds,
        panel_width_mm: float,
        panel_height_mm: float,
    ) -> np.ndarray:
        """Compute the luminance map on the diffuser plane.

        Parameters
        ----------
        leds:
            Iterable of :class:`~led_simulation.led.LED` objects.
        panel_width_mm:
            Physical width of the panel (mm).
        panel_height_mm:
            Physical height of the panel (mm).

        Returns
        -------
        numpy.ndarray
            2-D array of shape (height_px, width_px) containing relative
            luminance values in the range [0, 1].
        """
        width_px = max(1, int(round(panel_width_mm / self.resolution_mm)))
        height_px = max(1, int(round(panel_height_mm / self.resolution_mm)))

        luminance = np.zeros((height_px, width_px), dtype=np.float64)

        # Pixel-centre coordinate grids (mm)
        x_coords = np.linspace(0, panel_width_mm, width_px)
        y_coords = np.linspace(0, panel_height_mm, height_px)
        X, Y = np.meshgrid(x_coords, y_coords)

        for led in leds:
            if not led.enabled:
                continue
            sigma = self._sigma_from_half_angle(led.half_angle)
            kernel = self._gaussian(X, Y, led.x, led.y, sigma)
            luminance += led.effective_brightness * kernel

        # Normalise to [0, 1]
        peak = luminance.max()
        if peak > 0:
            luminance /= peak

        return luminance

    def compute_color_map(
        self,
        leds,
        panel_width_mm: float,
        panel_height_mm: float,
    ) -> np.ndarray:
        """Compute an RGB color map on the diffuser plane.

        Returns
        -------
        numpy.ndarray
            3-D array of shape (height_px, width_px, 3) with values in [0, 1].
        """
        width_px = max(1, int(round(panel_width_mm / self.resolution_mm)))
        height_px = max(1, int(round(panel_height_mm / self.resolution_mm)))

        color_map = np.zeros((height_px, width_px, 3), dtype=np.float64)

        x_coords = np.linspace(0, panel_width_mm, width_px)
        y_coords = np.linspace(0, panel_height_mm, height_px)
        X, Y = np.meshgrid(x_coords, y_coords)

        for led in leds:
            if not led.enabled:
                continue
            sigma = self._sigma_from_half_angle(led.half_angle)
            kernel = self._gaussian(X, Y, led.x, led.y, sigma)
            r, g, b = led.normalized_color
            color_map[:, :, 0] += led.effective_brightness * r * kernel
            color_map[:, :, 1] += led.effective_brightness * g * kernel
            color_map[:, :, 2] += led.effective_brightness * b * kernel

        # Normalise each channel independently
        peak = color_map.max()
        if peak > 0:
            color_map /= peak

        return np.clip(color_map, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Uniformity metrics
    # ------------------------------------------------------------------

    @staticmethod
    def uniformity(luminance_map: np.ndarray) -> float:
        """Return luminance uniformity as defined by SEMI D35.

        Uniformity = L_min / L_max × 100 (%)
        """
        lmax = luminance_map.max()
        lmin = luminance_map.min()
        if lmax == 0:
            return 0.0
        return float(lmin / lmax * 100.0)

    @staticmethod
    def rms_uniformity(luminance_map: np.ndarray) -> float:
        """Return RMS uniformity (%).

        RMS uniformity = (1 – σ/µ) × 100, where σ is the standard
        deviation and µ is the mean luminance.
        """
        mean = luminance_map.mean()
        if mean == 0:
            return 0.0
        std = luminance_map.std()
        return float((1.0 - std / mean) * 100.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sigma_from_half_angle(self, half_angle_deg: float) -> float:
        """Convert LED half-angle to Gaussian sigma (mm).

        The spread on the diffuser plane is approximated as
        sigma = OAD × tan(half_angle).
        """
        return self.oad_mm * np.tan(np.radians(half_angle_deg))

    @staticmethod
    def _gaussian(
        X: np.ndarray,
        Y: np.ndarray,
        cx: float,
        cy: float,
        sigma: float,
    ) -> np.ndarray:
        """Evaluate a 2-D symmetric Gaussian centred at (cx, cy)."""
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        return np.exp(-dist_sq / (2.0 * sigma ** 2))
