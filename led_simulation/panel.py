"""LED panel model representing a 2-D array of LEDs."""

from typing import List, Optional, Tuple

import numpy as np

from .led import LED


class LEDPanel:
    """A rectangular panel of LEDs arranged in a regular grid.

    Parameters
    ----------
    rows:
        Number of LED rows.
    cols:
        Number of LED columns.
    panel_width_mm:
        Physical width of the panel in millimetres.
    panel_height_mm:
        Physical height of the panel in millimetres.
    default_brightness:
        Initial brightness applied to every LED (0–1).
    default_color:
        Initial RGB color applied to every LED.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        panel_width_mm: float = 300.0,
        panel_height_mm: float = 200.0,
        default_brightness: float = 1.0,
        default_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if panel_width_mm <= 0 or panel_height_mm <= 0:
            raise ValueError("panel dimensions must be positive")

        self.rows = rows
        self.cols = cols
        self.panel_width_mm = panel_width_mm
        self.panel_height_mm = panel_height_mm

        self._leds: List[List[LED]] = []
        self._build_grid(default_brightness, default_color)

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _build_grid(
        self,
        default_brightness: float,
        default_color: Tuple[int, int, int],
    ) -> None:
        """Populate the LED grid with evenly-spaced LEDs."""
        x_positions = np.linspace(0, self.panel_width_mm, self.cols)
        y_positions = np.linspace(0, self.panel_height_mm, self.rows)

        self._leds = [
            [
                LED(
                    x=float(x_positions[c]),
                    y=float(y_positions[r]),
                    brightness=default_brightness,
                    color=default_color,
                )
                for c in range(self.cols)
            ]
            for r in range(self.rows)
        ]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_led(self, row: int, col: int) -> LED:
        """Return the LED at position (row, col)."""
        return self._leds[row][col]

    def all_leds(self) -> List[LED]:
        """Return a flat list of all LEDs on the panel."""
        return [led for row in self._leds for led in row]

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def set_brightness(self, row: int, col: int, brightness: float) -> None:
        """Set the brightness of a single LED."""
        self._leds[row][col].brightness = brightness

    def set_brightness_map(self, brightness_map: np.ndarray) -> None:
        """Apply a (rows × cols) numpy array of brightness values."""
        if brightness_map.shape != (self.rows, self.cols):
            raise ValueError(
                f"brightness_map shape {brightness_map.shape} does not match "
                f"panel shape ({self.rows}, {self.cols})"
            )
        for r in range(self.rows):
            for c in range(self.cols):
                self._leds[r][c].brightness = float(brightness_map[r, c])

    def set_zone_brightness(
        self,
        brightness: float,
        row_start: int = 0,
        row_end: Optional[int] = None,
        col_start: int = 0,
        col_end: Optional[int] = None,
    ) -> None:
        """Set uniform brightness for a rectangular zone of LEDs."""
        row_end = row_end if row_end is not None else self.rows
        col_end = col_end if col_end is not None else self.cols
        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                self._leds[r][c].brightness = brightness

    def disable_led(self, row: int, col: int) -> None:
        """Disable a single LED (simulates a failed LED)."""
        self._leds[row][col].enabled = False

    def enable_led(self, row: int, col: int) -> None:
        """Re-enable a previously disabled LED."""
        self._leds[row][col].enabled = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def led_count(self) -> int:
        return self.rows * self.cols

    @property
    def pitch_x_mm(self) -> float:
        """Horizontal pitch between adjacent LEDs (mm)."""
        if self.cols < 2:
            return self.panel_width_mm
        return self.panel_width_mm / (self.cols - 1)

    @property
    def pitch_y_mm(self) -> float:
        """Vertical pitch between adjacent LEDs (mm)."""
        if self.rows < 2:
            return self.panel_height_mm
        return self.panel_height_mm / (self.rows - 1)

    def brightness_array(self) -> np.ndarray:
        """Return a (rows × cols) numpy array of LED brightness values."""
        return np.array(
            [[self._leds[r][c].effective_brightness for c in range(self.cols)]
             for r in range(self.rows)]
        )

    def __repr__(self) -> str:
        return (
            f"LEDPanel(rows={self.rows}, cols={self.cols}, "
            f"size={self.panel_width_mm:.0f}x{self.panel_height_mm:.0f}mm)"
        )
