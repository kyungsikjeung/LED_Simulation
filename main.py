#!/usr/bin/env python3
"""LED Simulation for TOVIS — command-line entry point.

Usage examples
--------------
Run the default simulation and display plots:
    python main.py

Customise panel geometry:
    python main.py --rows 8 --cols 12 --width 400 --height 250

Increase OAD for better uniformity:
    python main.py --oad 50

Save output images to a directory:
    python main.py --output output/

Disable specific LEDs (simulate failures):
    python main.py --disable 2,3 --disable 4,5

Apply a local-dimming zone (centre zone at 50% brightness):
    python main.py --zone "2,10,2,6,0.5"
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for headless environments)

from led_simulation import LED, LEDPanel, Simulation
from led_simulation.visualization import (
    plot_brightness_map,
    plot_color_map,
    plot_cross_section,
    plot_luminance_map,
)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="led_simulation",
        description="LED backlight simulation for TOVIS display panels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Panel geometry
    p.add_argument("--rows", type=int, default=6,
                   help="Number of LED rows (default: 6)")
    p.add_argument("--cols", type=int, default=8,
                   help="Number of LED columns (default: 8)")
    p.add_argument("--width", type=float, default=300.0,
                   help="Panel width in mm (default: 300)")
    p.add_argument("--height", type=float, default=200.0,
                   help="Panel height in mm (default: 200)")

    # Optical parameters
    p.add_argument("--oad", type=float, default=30.0,
                   help="Optical Air Distance in mm (default: 30)")
    p.add_argument("--half-angle", type=float, default=60.0,
                   help="LED half-angle in degrees (default: 60)")
    p.add_argument("--resolution", type=float, default=1.0,
                   help="Map resolution in mm/pixel (default: 1.0)")

    # LED overrides
    p.add_argument("--disable", action="append", metavar="ROW,COL",
                   help="Disable LED at position ROW,COL (repeatable)")
    p.add_argument("--zone", action="append",
                   metavar="ROW_START,ROW_END,COL_START,COL_END,BRIGHTNESS",
                   help="Set brightness for a rectangular zone (repeatable)")

    # Output
    p.add_argument("--output", type=str, default=None,
                   help="Directory to save plot images (default: show interactively)")
    p.add_argument("--no-show", action="store_true",
                   help="Do not call plt.show() (useful in CI/headless mode)")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress printed output")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    show_plots = not args.no_show and args.output is None

    # If we are not saving and not showing, default to saving in ./output/
    if not show_plots and args.output is None:
        args.output = "output"

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    # Build panel
    panel = LEDPanel(
        rows=args.rows,
        cols=args.cols,
        panel_width_mm=args.width,
        panel_height_mm=args.height,
    )

    # Apply per-LED half-angle
    for led in panel.all_leds():
        led.half_angle = args.half_angle

    # Disable individual LEDs
    if args.disable:
        for spec in args.disable:
            try:
                r, c = [int(x.strip()) for x in spec.split(",")]
                panel.disable_led(r, c)
            except (ValueError, IndexError):
                print(f"[WARNING] Invalid --disable spec '{spec}', skipping.",
                      file=sys.stderr)

    # Apply brightness zones
    if args.zone:
        for spec in args.zone:
            parts = [x.strip() for x in spec.split(",")]
            if len(parts) != 5:
                print(f"[WARNING] Invalid --zone spec '{spec}', skipping.",
                      file=sys.stderr)
                continue
            try:
                r0, r1, c0, c1 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                brightness = float(parts[4])
                panel.set_zone_brightness(brightness, r0, r1, c0, c1)
            except ValueError:
                print(f"[WARNING] Invalid --zone spec '{spec}', skipping.",
                      file=sys.stderr)

    # Run simulation
    sim = Simulation(panel, oad_mm=args.oad, resolution_mm=args.resolution)
    result = sim.run(notes=f"rows={args.rows}, cols={args.cols}, oad={args.oad}mm")

    if not args.quiet:
        print("=" * 50)
        print("  LED Simulation for TOVIS")
        print("=" * 50)
        print(result.summary())
        print("=" * 50)

    # Determine save paths
    def _path(filename: str) -> str:
        if args.output:
            return str(Path(args.output) / filename)
        return None

    # Plot 1: Luminance heat-map
    plot_luminance_map(
        result,
        panel=panel,
        title=f"Luminance Distribution — {args.rows}×{args.cols} LEDs, OAD={args.oad}mm",
        save_path=_path("luminance_map.png"),
        show=show_plots,
    )

    # Plot 2: Brightness map of the LED array
    plot_brightness_map(
        panel,
        title="LED Brightness Map",
        save_path=_path("brightness_map.png"),
        show=show_plots,
    )

    # Plot 3: Horizontal cross-section
    plot_cross_section(
        result,
        axis="x",
        save_path=_path("cross_section_x.png"),
        show=show_plots,
    )

    # Plot 4: Vertical cross-section
    plot_cross_section(
        result,
        axis="y",
        save_path=_path("cross_section_y.png"),
        show=show_plots,
    )

    if args.output and not args.quiet:
        print(f"\nPlots saved to: {os.path.abspath(args.output)}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
