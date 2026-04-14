"""LED model representing a single LED light source."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class LED:
    """Represents a single LED in the backlight panel.

    Attributes:
        x: Horizontal position of the LED (mm).
        y: Vertical position of the LED (mm).
        brightness: Relative luminous intensity in the range [0.0, 1.0].
        color: RGB color tuple with values in the range [0, 255].
        half_angle: Half-angle of the LED emission cone in degrees.
        enabled: Whether the LED is active.
    """

    x: float
    y: float
    brightness: float = 1.0
    color: Tuple[int, int, int] = field(default_factory=lambda: (255, 255, 255))
    half_angle: float = 60.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.brightness <= 1.0:
            raise ValueError(
                f"brightness must be in [0, 1], got {self.brightness}"
            )
        if len(self.color) != 3 or not all(0 <= c <= 255 for c in self.color):
            raise ValueError(
                f"color must be an (R, G, B) tuple with values in [0, 255], got {self.color}"
            )
        if not 0.0 < self.half_angle <= 90.0:
            raise ValueError(
                f"half_angle must be in (0, 90], got {self.half_angle}"
            )

    @property
    def normalized_color(self) -> Tuple[float, float, float]:
        """Return RGB color normalized to [0.0, 1.0]."""
        return tuple(c / 255.0 for c in self.color)

    @property
    def effective_brightness(self) -> float:
        """Return effective brightness (0.0 when disabled)."""
        return self.brightness if self.enabled else 0.0

    def __repr__(self) -> str:
        return (
            f"LED(x={self.x:.2f}, y={self.y:.2f}, "
            f"brightness={self.brightness:.2f}, enabled={self.enabled})"
        )
