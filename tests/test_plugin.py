"""Tests for ScreenVision.

Run: `pytest`.

Level 1: in process, no daemon, no socket, fast enough to run on every save.
Still goes through the real gRPC servicer, so a tool that is declared but not
routed fails here.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astra_plugin_sdk.testing import Harness, fuzz_configs  # noqa: E402

from src.plugin import ScreenVision  # noqa: E402


def test_tool_names_and_schema():
    with Harness(ScreenVision()) as h:
        assert h.tool_names() == ["look_at_screen"]
        h.assert_schema_accepts("look_at_screen", "question", "monitor")
        assert h.schema("look_at_screen")["required"] == ["question"]


def test_missing_base_url_is_not_configured():
    with Harness(ScreenVision()) as h:
        h.plugin._settings = {"base_url": "", "api_key": "", "model": ""}
        result = h.call_tool("look_at_screen", question="what is on screen?")
        assert not result.success
        assert result.code == "NOT_CONFIGURED"


def test_missing_model_is_not_configured():
    with Harness(ScreenVision()) as h:
        h.plugin._settings = {"base_url": "http://localhost:11434/v1", "api_key": "", "model": ""}
        result = h.call_tool("look_at_screen", question="what is on screen?")
        assert not result.success
        assert result.code == "NOT_CONFIGURED"


def test_get_settings_returns_values():
    with Harness(ScreenVision()) as h:
        h.plugin._settings = {"base_url": "http://x/v1", "api_key": "k", "model": "m"}
        result = h.ui_call("get_settings")
        assert result.json == {"base_url": "http://x/v1", "api_key": "k", "model": "m"}


def test_save_settings_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(ScreenVision, "_settings_path", lambda self: tmp_path / "settings.json")
    with Harness(ScreenVision()) as h:
        h.plugin._settings = {"base_url": "", "api_key": "", "model": ""}
        result = h.ui_call("save_settings", base_url="http://x/v1", api_key="k", model="m")
        assert result.json["ok"] is True
        assert h.plugin._settings == {"base_url": "http://x/v1", "api_key": "k", "model": "m"}
        saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
        assert saved["base_url"] == "http://x/v1"


def test_settings_panel_is_transparent():
    with Harness(ScreenVision()) as h:
        contribs = h.run(h.plugin.get_ui_contributions())
        assert len(contribs) == 1
        contribution = contribs[0]
        assert contribution.slot == "settings.custom"
        assert contribution.url == "settings.html"
        assert contribution.transparent is True


def test_no_config_the_daemon_can_deliver_crashes_this_plugin():
    with Harness(ScreenVision()) as h:
        for payload in fuzz_configs():
            h.set_config(payload)
