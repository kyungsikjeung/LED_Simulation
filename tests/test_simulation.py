"""Tests for the LED Simulation package."""

import numpy as np
import pytest

from led_simulation.led import LED
from led_simulation.panel import LEDPanel
from led_simulation.diffusion import DiffusionModel
from led_simulation.simulation import Simulation, SimulationResult


# ---------------------------------------------------------------------------
# LED tests
# ---------------------------------------------------------------------------

class TestLED:
    def test_default_values(self):
        led = LED(x=10.0, y=20.0)
        assert led.brightness == 1.0
        assert led.color == (255, 255, 255)
        assert led.half_angle == 60.0
        assert led.enabled is True

    def test_effective_brightness_when_enabled(self):
        led = LED(x=0, y=0, brightness=0.7)
        assert led.effective_brightness == pytest.approx(0.7)

    def test_effective_brightness_when_disabled(self):
        led = LED(x=0, y=0, brightness=0.9, enabled=False)
        assert led.effective_brightness == 0.0

    def test_normalized_color(self):
        led = LED(x=0, y=0, color=(255, 128, 0))
        r, g, b = led.normalized_color
        assert r == pytest.approx(1.0)
        assert g == pytest.approx(128 / 255)
        assert b == pytest.approx(0.0)

    def test_invalid_brightness_raises(self):
        with pytest.raises(ValueError, match="brightness"):
            LED(x=0, y=0, brightness=1.5)

    def test_invalid_color_raises(self):
        with pytest.raises(ValueError, match="color"):
            LED(x=0, y=0, color=(256, 0, 0))

    def test_invalid_half_angle_raises(self):
        with pytest.raises(ValueError, match="half_angle"):
            LED(x=0, y=0, half_angle=0.0)


# ---------------------------------------------------------------------------
# LEDPanel tests
# ---------------------------------------------------------------------------

class TestLEDPanel:
    def test_panel_has_correct_led_count(self):
        panel = LEDPanel(rows=4, cols=5)
        assert panel.led_count == 20
        assert len(panel.all_leds()) == 20

    def test_led_positions_span_panel(self):
        panel = LEDPanel(rows=3, cols=3, panel_width_mm=100.0, panel_height_mm=100.0)
        xs = [led.x for led in panel.all_leds()]
        ys = [led.y for led in panel.all_leds()]
        assert min(xs) == pytest.approx(0.0)
        assert max(xs) == pytest.approx(100.0)
        assert min(ys) == pytest.approx(0.0)
        assert max(ys) == pytest.approx(100.0)

    def test_set_brightness(self):
        panel = LEDPanel(rows=3, cols=3)
        panel.set_brightness(1, 1, 0.5)
        assert panel.get_led(1, 1).brightness == pytest.approx(0.5)

    def test_set_brightness_map(self):
        panel = LEDPanel(rows=2, cols=2)
        bmap = np.array([[0.1, 0.2], [0.3, 0.4]])
        panel.set_brightness_map(bmap)
        assert panel.get_led(0, 0).brightness == pytest.approx(0.1)
        assert panel.get_led(1, 1).brightness == pytest.approx(0.4)

    def test_set_brightness_map_wrong_shape_raises(self):
        panel = LEDPanel(rows=2, cols=2)
        with pytest.raises(ValueError, match="shape"):
            panel.set_brightness_map(np.ones((3, 3)))

    def test_disable_enable_led(self):
        panel = LEDPanel(rows=3, cols=3)
        panel.disable_led(0, 0)
        assert panel.get_led(0, 0).enabled is False
        panel.enable_led(0, 0)
        assert panel.get_led(0, 0).enabled is True

    def test_zone_brightness(self):
        panel = LEDPanel(rows=4, cols=4)
        panel.set_zone_brightness(0.3, row_start=1, row_end=3, col_start=1, col_end=3)
        assert panel.get_led(1, 1).brightness == pytest.approx(0.3)
        assert panel.get_led(0, 0).brightness == pytest.approx(1.0)

    def test_brightness_array_shape(self):
        panel = LEDPanel(rows=3, cols=5)
        barray = panel.brightness_array()
        assert barray.shape == (3, 5)

    def test_pitch_values(self):
        panel = LEDPanel(rows=3, cols=5, panel_width_mm=80.0, panel_height_mm=40.0)
        assert panel.pitch_x_mm == pytest.approx(20.0)
        assert panel.pitch_y_mm == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# DiffusionModel tests
# ---------------------------------------------------------------------------

class TestDiffusionModel:
    def test_luminance_map_shape(self):
        panel = LEDPanel(rows=3, cols=3, panel_width_mm=30.0, panel_height_mm=30.0)
        model = DiffusionModel(oad_mm=20.0, resolution_mm=1.0)
        lmap = model.compute_luminance_map(
            panel.all_leds(), panel.panel_width_mm, panel.panel_height_mm
        )
        assert lmap.shape == (30, 30)

    def test_luminance_map_range(self):
        panel = LEDPanel(rows=3, cols=3, panel_width_mm=60.0, panel_height_mm=60.0)
        model = DiffusionModel(oad_mm=20.0, resolution_mm=1.0)
        lmap = model.compute_luminance_map(
            panel.all_leds(), panel.panel_width_mm, panel.panel_height_mm
        )
        assert lmap.min() >= 0.0
        assert lmap.max() <= 1.0 + 1e-9

    def test_disabled_led_does_not_contribute(self):
        from led_simulation.led import LED
        led_on = LED(x=15.0, y=15.0, brightness=1.0)
        led_off = LED(x=15.0, y=15.0, brightness=1.0, enabled=False)
        model = DiffusionModel(oad_mm=20.0, resolution_mm=1.0)

        lmap_on = model.compute_luminance_map([led_on], 30.0, 30.0)
        lmap_off = model.compute_luminance_map([led_off], 30.0, 30.0)
        assert lmap_off.max() == pytest.approx(0.0)
        assert lmap_on.max() > 0.0

    def test_uniformity_perfect(self):
        uniform = np.ones((10, 10))
        assert DiffusionModel.uniformity(uniform) == pytest.approx(100.0)

    def test_uniformity_zero_map(self):
        assert DiffusionModel.uniformity(np.zeros((5, 5))) == pytest.approx(0.0)

    def test_rms_uniformity_perfect(self):
        uniform = np.ones((10, 10))
        assert DiffusionModel.rms_uniformity(uniform) == pytest.approx(100.0)

    def test_color_map_shape_and_range(self):
        panel = LEDPanel(rows=2, cols=2, panel_width_mm=20.0, panel_height_mm=20.0)
        model = DiffusionModel(oad_mm=15.0, resolution_mm=1.0)
        cmap = model.compute_color_map(
            panel.all_leds(), panel.panel_width_mm, panel.panel_height_mm
        )
        assert cmap.shape == (20, 20, 3)
        assert cmap.min() >= 0.0
        assert cmap.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_run_returns_result(self):
        panel = LEDPanel(rows=3, cols=4, panel_width_mm=60.0, panel_height_mm=45.0)
        sim = Simulation(panel, oad_mm=20.0, resolution_mm=1.0)
        result = sim.run()
        assert isinstance(result, SimulationResult)

    def test_result_uniformity_in_range(self):
        panel = LEDPanel(rows=4, cols=6, panel_width_mm=120.0, panel_height_mm=80.0)
        sim = Simulation(panel, oad_mm=30.0, resolution_mm=2.0)
        result = sim.run()
        assert 0.0 <= result.uniformity_pct <= 100.0
        assert 0.0 <= result.rms_uniformity_pct <= 100.0

    def test_higher_oad_improves_uniformity(self):
        panel_low = LEDPanel(rows=4, cols=6, panel_width_mm=120.0, panel_height_mm=80.0)
        panel_high = LEDPanel(rows=4, cols=6, panel_width_mm=120.0, panel_height_mm=80.0)
        result_low = Simulation(panel_low, oad_mm=10.0, resolution_mm=2.0).run()
        result_high = Simulation(panel_high, oad_mm=80.0, resolution_mm=2.0).run()
        assert result_high.uniformity_pct > result_low.uniformity_pct

    def test_quick_run(self):
        result = Simulation.quick_run(rows=4, cols=4, panel_width_mm=80.0,
                                      panel_height_mm=80.0, oad_mm=20.0,
                                      resolution_mm=2.0)
        assert result.luminance_map.max() == pytest.approx(1.0, abs=1e-9)

    def test_result_summary_contains_uniformity(self):
        result = Simulation.quick_run(rows=3, cols=3, resolution_mm=2.0)
        summary = result.summary()
        assert "Uniformity" in summary

    def test_disabled_led_lowers_luminance(self):
        from led_simulation.diffusion import DiffusionModel
        panel_full = LEDPanel(rows=3, cols=3, panel_width_mm=60.0, panel_height_mm=60.0)
        panel_partial = LEDPanel(rows=3, cols=3, panel_width_mm=60.0, panel_height_mm=60.0)
        panel_partial.disable_led(1, 1)

        model = DiffusionModel(oad_mm=20.0, resolution_mm=2.0)

        # Compute raw (unnormalised) luminance maps to compare energy
        def raw_luminance(p):
            from led_simulation.diffusion import DiffusionModel as DM
            import numpy as _np
            m = DM(oad_mm=20.0, resolution_mm=2.0)
            w_px = max(1, int(round(p.panel_width_mm / m.resolution_mm)))
            h_px = max(1, int(round(p.panel_height_mm / m.resolution_mm)))
            x_coords = _np.linspace(0, p.panel_width_mm, w_px)
            y_coords = _np.linspace(0, p.panel_height_mm, h_px)
            X, Y = _np.meshgrid(x_coords, y_coords)
            lum = _np.zeros((h_px, w_px))
            for led in p.all_leds():
                if not led.enabled:
                    continue
                sigma = m._sigma_from_half_angle(led.half_angle)
                lum += led.effective_brightness * m._gaussian(X, Y, led.x, led.y, sigma)
            return lum

        lum_full = raw_luminance(panel_full)
        lum_partial = raw_luminance(panel_partial)

        # Disabling the centre LED should reduce total raw luminance energy
        assert lum_partial.sum() < lum_full.sum()
