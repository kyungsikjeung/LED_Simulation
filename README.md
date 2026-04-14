# LED Simulation for TOVIS

A Python-based LED backlight simulation tool for TOVIS display panels.

The simulation models how an array of LEDs illuminates a diffuser plate using
a 2-D Gaussian light-diffusion model, and reports luminance uniformity metrics
(SEMI D35 and RMS uniformity).

---

## Features

| Feature | Description |
|---|---|
| LED matrix | Configurable rows × cols LED array on a physical panel |
| Gaussian diffusion | Physically-motivated light spread based on LED half-angle and optical air distance (OAD) |
| Local dimming | Set per-LED or zone-based brightness values |
| Fault simulation | Disable individual LEDs to simulate failures |
| Uniformity metrics | SEMI D35 and RMS uniformity percentages |
| Visualisation | Luminance heat-map, brightness map, and X/Y cross-section plots |
| CLI | Fully parametric command-line interface |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

Run the default simulation (6 × 8 LEDs, 300 × 200 mm panel, 30 mm OAD):

```bash
python main.py
```

Save output plots to a directory:

```bash
python main.py --output output/
```

---

## Command-Line Options

```
usage: led_simulation [-h] [--rows ROWS] [--cols COLS] [--width WIDTH]
                      [--height HEIGHT] [--oad OAD] [--half-angle HALF_ANGLE]
                      [--resolution RESOLUTION] [--disable ROW,COL]
                      [--zone ROW_START,ROW_END,COL_START,COL_END,BRIGHTNESS]
                      [--output OUTPUT] [--no-show] [--quiet]

Panel geometry:
  --rows ROWS          Number of LED rows (default: 6)
  --cols COLS          Number of LED columns (default: 8)
  --width WIDTH        Panel width in mm (default: 300)
  --height HEIGHT      Panel height in mm (default: 200)

Optical parameters:
  --oad OAD            Optical Air Distance in mm (default: 30)
  --half-angle DEG     LED half-angle in degrees (default: 60)
  --resolution MM/PX   Map resolution in mm/pixel (default: 1.0)

LED overrides:
  --disable ROW,COL    Disable LED at ROW,COL (repeatable)
  --zone R0,R1,C0,C1,B Set brightness B for zone rows R0–R1, cols C0–C1

Output:
  --output DIR         Directory to save PNG plots
  --no-show            Skip plt.show() (headless / CI use)
  --quiet, -q          Suppress printed summary
```

### Examples

Larger panel with increased OAD for better uniformity:

```bash
python main.py --rows 8 --cols 12 --width 400 --height 250 --oad 50
```

Simulate two failed LEDs:

```bash
python main.py --disable 2,3 --disable 4,5
```

Apply a local-dimming zone (rows 1–3, cols 2–4 at 30% brightness):

```bash
python main.py --zone "1,3,2,4,0.3"
```

---

## Python API

```python
from led_simulation import LEDPanel, Simulation

# Create a 6×8 LED panel, 300 × 200 mm
panel = LEDPanel(rows=6, cols=8, panel_width_mm=300, panel_height_mm=200)

# Disable a failed LED
panel.disable_led(2, 3)

# Apply local dimming to a zone
panel.set_zone_brightness(0.5, row_start=0, row_end=3, col_start=0, col_end=4)

# Run the simulation (OAD = 30 mm)
sim = Simulation(panel, oad_mm=30.0)
result = sim.run()

print(result.summary())
# Panel  : 300 × 200 mm
# OAD    : 30.0 mm
# ...
# Uniformity (SEMI D35): XX.XX%
# RMS Uniformity       : XX.XX%

# Visualise
from led_simulation.visualization import plot_luminance_map
plot_luminance_map(result, panel=panel, save_path="luminance.png", show=False)
```

---

## Project Structure

```
LED_Simulation/
├── led_simulation/          # Core package
│   ├── __init__.py
│   ├── led.py               # LED data model
│   ├── panel.py             # LED panel / matrix
│   ├── diffusion.py         # Gaussian diffusion model + uniformity metrics
│   ├── simulation.py        # Simulation engine
│   └── visualization.py     # Matplotlib visualisation helpers
├── tests/
│   └── test_simulation.py   # Unit tests (pytest)
├── main.py                  # CLI entry point
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```
