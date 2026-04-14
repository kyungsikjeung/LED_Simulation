"""Visualization helpers for LED simulation results."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from .panel import LEDPanel
from .simulation import SimulationResult


def plot_luminance_map(
    result: SimulationResult,
    panel: Optional[LEDPanel] = None,
    title: str = "Luminance Distribution",
    show_leds: bool = True,
    colormap: str = "hot",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the luminance distribution heat-map.

    Parameters
    ----------
    result:
        :class:`~led_simulation.simulation.SimulationResult` to visualize.
    panel:
        When provided, the LED positions are overlaid on the heat-map.
    title:
        Plot title.
    show_leds:
        If *True* (default) and *panel* is given, draw LED positions.
    colormap:
        Matplotlib colormap name.
    save_path:
        If given, the figure is saved to this file path.
    show:
        If *True* (default) ``plt.show()`` is called.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    extent = [0, result.panel_width_mm, 0, result.panel_height_mm]
    im = ax.imshow(
        result.luminance_map,
        origin="lower",
        extent=extent,
        cmap=colormap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Luminance", fontsize=11)

    if show_leds and panel is not None:
        xs = [led.x for led in panel.all_leds() if led.enabled]
        ys = [led.y for led in panel.all_leds() if led.enabled]
        xs_off = [led.x for led in panel.all_leds() if not led.enabled]
        ys_off = [led.y for led in panel.all_leds() if not led.enabled]
        ax.scatter(xs, ys, c="white", s=30, marker="o",
                   linewidths=0.8, edgecolors="gray", label="LED (on)", zorder=5)
        if xs_off:
            ax.scatter(xs_off, ys_off, c="black", s=30, marker="x",
                       linewidths=1.2, label="LED (off)", zorder=5)
        ax.legend(loc="upper right", fontsize=9)

    ax.set_xlabel("X (mm)", fontsize=11)
    ax.set_ylabel("Y (mm)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

    stats_text = (
        f"Uniformity: {result.uniformity_pct:.1f}%\n"
        f"RMS Uniformity: {result.rms_uniformity_pct:.1f}%"
    )
    ax.text(
        0.02, 0.03, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_color_map(
    result: SimulationResult,
    panel: Optional[LEDPanel] = None,
    title: str = "Color Distribution",
    show_leds: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the RGB color distribution on the diffuser plane.

    Parameters
    ----------
    result:
        :class:`~led_simulation.simulation.SimulationResult` to visualize.
    panel:
        When provided, LED positions are overlaid.
    title:
        Plot title.
    show_leds:
        If *True* and *panel* is given, draw LED positions.
    save_path:
        Optional file path to save the figure.
    show:
        Call ``plt.show()`` when *True*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    extent = [0, result.panel_width_mm, 0, result.panel_height_mm]
    ax.imshow(
        result.color_map,
        origin="lower",
        extent=extent,
        aspect="auto",
    )

    if show_leds and panel is not None:
        xs = [led.x for led in panel.all_leds() if led.enabled]
        ys = [led.y for led in panel.all_leds() if led.enabled]
        ax.scatter(xs, ys, c="white", s=30, marker="o",
                   linewidths=0.8, edgecolors="gray", zorder=5)

    ax.set_xlabel("X (mm)", fontsize=11)
    ax.set_ylabel("Y (mm)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_brightness_map(
    panel: LEDPanel,
    title: str = "LED Brightness Map",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the current brightness of each LED as a heat-map grid.

    Parameters
    ----------
    panel:
        The :class:`~led_simulation.panel.LEDPanel` to visualize.
    title:
        Plot title.
    save_path:
        Optional file path to save the figure.
    show:
        Call ``plt.show()`` when *True*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bmap = panel.brightness_array()
    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(
        bmap,
        origin="upper",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Brightness", fontsize=11)

    ax.set_xlabel("Column", fontsize=11)
    ax.set_ylabel("Row", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.set_xticks(range(panel.cols))
    ax.set_yticks(range(panel.rows))

    # Annotate each cell with the brightness value
    for r in range(panel.rows):
        for c in range(panel.cols):
            val = bmap[r, c]
            color = "black" if val > 0.5 else "white"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_cross_section(
    result: SimulationResult,
    axis: str = "x",
    position_mm: Optional[float] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot the luminance cross-section along the X or Y axis.

    Parameters
    ----------
    result:
        Simulation result to analyse.
    axis:
        ``'x'`` for a horizontal slice, ``'y'`` for a vertical slice.
    position_mm:
        Position of the slice along the perpendicular axis (mm).
        Defaults to the centre of the panel.
    title:
        Plot title.  Auto-generated when *None*.
    save_path:
        Optional file path to save the figure.
    show:
        Call ``plt.show()`` when *True*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    lmap = result.luminance_map
    h_px, w_px = lmap.shape

    if axis.lower() == "x":
        if position_mm is None:
            position_mm = result.panel_height_mm / 2.0
        row = int(round(position_mm / result.resolution_mm))
        row = max(0, min(row, h_px - 1))
        profile = lmap[row, :]
        coords = np.linspace(0, result.panel_width_mm, w_px)
        xlabel = "X (mm)"
        default_title = f"Horizontal Cross-section at Y = {position_mm:.0f} mm"
    else:
        if position_mm is None:
            position_mm = result.panel_width_mm / 2.0
        col = int(round(position_mm / result.resolution_mm))
        col = max(0, min(col, w_px - 1))
        profile = lmap[:, col]
        coords = np.linspace(0, result.panel_height_mm, h_px)
        xlabel = "Y (mm)"
        default_title = f"Vertical Cross-section at X = {position_mm:.0f} mm"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(coords, profile, linewidth=1.8, color="royalblue")
    ax.fill_between(coords, profile, alpha=0.25, color="royalblue")
    ax.axhline(profile.mean(), color="red", linestyle="--",
               linewidth=1.2, label=f"Mean: {profile.mean():.3f}")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Normalized Luminance", fontsize=11)
    ax.set_title(title or default_title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
