"""Visual regression testing configuration.

Defines pages and viewports for Playwright-based screenshot comparison.
Run with: pytest tests/visual/ --visual
"""

import pytest


VISUAL_TEST_PAGES = [
    {"name": "dashboard", "path": "/", "wait_for": "[data-testid='dashboard']"},
    {"name": "trading", "path": "/trading", "wait_for": "[data-testid='trading-page']"},
    {"name": "portfolio", "path": "/portfolio", "wait_for": "[data-testid='portfolio-page']"},
    {"name": "backtest", "path": "/backtest", "wait_for": "[data-testid='backtest-page']"},
    {"name": "signals", "path": "/signals", "wait_for": "[data-testid='signals-page']"},
    {"name": "settings", "path": "/settings", "wait_for": "[data-testid='settings-page']"},
    {"name": "alerts", "path": "/alerts", "wait_for": "main"},
    {"name": "sentiment", "path": "/sentiment", "wait_for": "main"},
    {"name": "journal", "path": "/trading/journal", "wait_for": "main"},
]

VIEWPORTS = [
    {"name": "desktop", "width": 1920, "height": 1080},
    {"name": "tablet", "width": 768, "height": 1024},
    {"name": "mobile", "width": 375, "height": 812},
]

SCREENSHOT_DIR = "tests/visual/screenshots"
BASELINE_DIR = "tests/visual/baselines"
DIFF_THRESHOLD = 0.001  # 0.1% pixel difference threshold


class TestVisualRegressionConfig:
    """Validate visual regression test configuration."""

    def test_all_pages_have_names(self):
        for page in VISUAL_TEST_PAGES:
            assert "name" in page
            assert "path" in page
            assert len(page["name"]) > 0

    def test_all_viewports_valid(self):
        for vp in VIEWPORTS:
            assert vp["width"] > 0
            assert vp["height"] > 0

    def test_diff_threshold_reasonable(self):
        assert 0 < DIFF_THRESHOLD < 0.01  # Max 1% difference

    def test_screenshot_dirs_defined(self):
        assert SCREENSHOT_DIR
        assert BASELINE_DIR
