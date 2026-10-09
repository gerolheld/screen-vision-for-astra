"""ScreenVision — give Astra eyes.

A single tool that captures the screen and answers a question about what is
visible on it, using any OpenAI-compatible vision model (cloud or local).

Settings live in `settings.json` next to the plugin, edited from a custom panel
in Astra's Settings (`settings.custom`).
"""

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
from PIL import Image

from astra_plugin_sdk import (
    InternalError,
    NotConfigured,
    Plugin,
    UiContribution,
    Unauthorized,
    tool,
    ui_call,
)

try:
    import mss
except ImportError:  # pragma: no cover - requirements.txt installs it
    mss = None

DEFAULT_SETTINGS = {
    "base_url": "",
    "api_key": "",
    "model": "",
}


class ScreenVision(Plugin):
    """Astra plugin: screen-vision."""

    def __init__(self):
        super().__init__()
        self._settings = dict(DEFAULT_SETTINGS)
        self._load_settings()

    # ── settings persistence ──

    def _settings_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "settings.json"

    def _load_settings(self) -> None:
        try:
            data = json.loads(self._settings_path().read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._settings.update(
                    {k: str(v) for k, v in data.items() if k in DEFAULT_SETTINGS}
                )
        except (OSError, ValueError):
            pass

    def _save_settings(self, settings: dict) -> None:
        for key in DEFAULT_SETTINGS:
            if key in settings:
                self._settings[key] = str(settings[key] or "").strip()
        self._settings_path().write_text(
            json.dumps(self._settings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── settings panel ──

    async def get_ui_contributions(self) -> list[UiContribution]:
        return [
            UiContribution(
                id="screen-vision-settings",
                slot="settings.custom",
                url="settings.html",
                label="Screen Vision",
                width=0,
                height=360,
                transparent=True,
                pointer_events=True,
            )
        ]

    @ui_call
    async def get_settings(self):
        return dict(self._settings)

    @ui_call
    async def save_settings(self, base_url: str = "", api_key: str = "", model: str = ""):
        try:
            self._save_settings({"base_url": base_url, "api_key": api_key, "model": model})
        except OSError as e:
            return {"ok": False, "error": f"Не удалось сохранить настройки: {e}"}
        return {"ok": True}

    @ui_call
    async def test_connection(self, base_url: str = "", api_key: str = "", model: str = ""):
        base_url = (base_url or "").strip()
        model = (model or "").strip()
        if not base_url:
            return {"ok": False, "error": "Задайте base_url"}
        if not model:
            return {"ok": False, "error": "Задайте model"}

        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        tiny_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with exactly: OK"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{tiny_png}"},
                        },
                    ],
                }
            ],
            "max_tokens": 8,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as e:
            return {"ok": False, "error": f"Не удалось подключиться: {e}"}

        if resp.status_code in (401, 403):
            return {"ok": False, "error": "API отверг ключ (401/403)"}
        if resp.status_code >= 400:
            body = resp.text[:400]
            if any(w in body.lower() for w in ("image", "vision", "content", "multimodal", "model")):
                return {
                    "ok": False,
                    "error": f"Модель '{model}' не принимает изображения. "
                    "Укажите vision-модель (gpt-4o, или локальную llava/qwen2.5-vl). "
                    f"Ответ API: {body}",
                }
            return {"ok": False, "error": f"API вернул {resp.status_code}: {body}"}
        return {"ok": True, "message": f"OK: модель '{model}' принимает изображения"}

    # ── the tool ──

    @tool(
        "Take a screenshot of the user's screen and answer a question about what "
        "is visible on it. Use whenever the user asks about something they can "
        "see or are looking at, for example: 'what car is this?', 'what does "
        "this error say?', 'read this page for me'."
    )
    async def look_at_screen(self, question: str, monitor: int = 1):
        base_url = self._settings.get("base_url") or ""
        api_key = self._settings.get("api_key") or ""
        model = self._settings.get("model") or ""

        if not base_url:
            raise NotConfigured("base_url", "Задайте base_url в настройках Screen Vision")
        if not model:
            raise NotConfigured("model", "Задайте model в настройках Screen Vision")

        image_b64 = self._capture(monitor)

        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a vision assistant. Answer the user's "
                    "question based on the screenshot they just took.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=110.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as e:
            raise InternalError(f"Could not reach the vision API at {base_url}: {e}")

        if resp.status_code in (401, 403):
            raise Unauthorized("The vision API rejected the configured API key")
        if resp.status_code >= 400:
            body = resp.text[:500]
            if any(w in body.lower() for w in ("image", "vision", "content", "multimodal")):
                raise InternalError(
                    f"The model '{model}' does not accept images. Use a vision-capable "
                    f"model (gpt-4o, or a local llava/qwen2.5-vl via Ollama). API said: {body}"
                )
            raise InternalError(f"Vision API returned {resp.status_code}: {body}")

        try:
            data = resp.json()
        except ValueError as e:
            raise InternalError(f"Vision API returned non-JSON: {e}")

        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise InternalError(
                "Unexpected vision API response: " + json.dumps(data)[:500]
            )

        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            content = "\n".join(parts)

        return str(content).strip()

    def _ensure_display_env(self) -> str | None:
        if not sys.platform.startswith("linux"):
            return None
        display = os.environ.get("DISPLAY") or ""
        if not display:
            x11 = Path("/tmp/.X11-unix")
            if x11.is_dir():
                names = sorted(p.name for p in x11.iterdir() if p.name.startswith("X"))
                if names:
                    display = ":" + names[0][1:]
        display = display or ":0"
        os.environ["DISPLAY"] = display

        if not os.environ.get("XAUTHORITY"):
            candidates = [Path.home() / ".Xauthority"]
            run_dir = Path(f"/run/user/{os.getuid()}")
            if run_dir.is_dir():
                candidates += sorted(run_dir.glob("xauth*"))
            for candidate in candidates:
                if candidate.is_file():
                    os.environ["XAUTHORITY"] = str(candidate)
                    break
        return display

    def _on_wayland(self) -> bool:
        if not sys.platform.startswith("linux"):
            return False
        run_dir = Path(f"/run/user/{os.getuid()}")
        if not run_dir.is_dir():
            return False
        return any(p.is_socket() for p in run_dir.glob("wayland-*"))

    def _ensure_wayland_env(self) -> bool:
        if not sys.platform.startswith("linux"):
            return False
        run_dir = Path(f"/run/user/{os.getuid()}")
        sockets = [p for p in run_dir.glob("wayland-*") if p.is_socket()]
        if not sockets:
            return False
        os.environ.setdefault("XDG_RUNTIME_DIR", str(run_dir))
        os.environ.setdefault("WAYLAND_DISPLAY", sorted(sockets)[0].name)
        bus = run_dir / "bus"
        if bus.exists():
            os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={bus}")
        return True

    def _capture_wayland(self) -> str | None:
        if not self._ensure_wayland_env():
            return None

        tools = [
            ["spectacle", "-b", "-n", "-o", "{path}"],
            ["gnome-screenshot", "-f", "{path}"],
            ["grim", "{path}"],
        ]
        for tool in tools:
            if shutil.which(tool[0]) is None:
                continue
            path = ""
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    path = tmp.name
                cmd = [arg.replace("{path}", path) for arg in tool]
                try:
                    proc = subprocess.run(
                        cmd, capture_output=True, timeout=20
                    )
                except (OSError, subprocess.TimeoutExpired):
                    continue
                if proc.returncode == 0 and Path(path).exists() and Path(path).stat().st_size > 0:
                    data = Path(path).read_bytes()
                    if not any(data):
                        continue
                    return base64.b64encode(data).decode("ascii")
            finally:
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
        return None

    def _capture_x11(self, monitor: int) -> str:
        if mss is None:
            raise InternalError("Neither a Wayland screenshot tool nor the `mss` package is available")

        display = self._ensure_display_env()
        try:
            kwargs = {"display": display} if display else {}
            with mss.MSS(**kwargs) as sct:
                monitors = sct.monitors
                index = monitor if 1 <= monitor < len(monitors) else 1
                shot = sct.grab(monitors[index])
        except Exception as e:
            raise InternalError(
                f"Could not capture the screen: {e}. "
                "On Wayland, screen capture may be unavailable — an X11 (or XWayland) "
                "session is required."
            )

        if not any(shot.rgb):
            raise InternalError(
                "Screen capture returned a black frame. On Wayland, X11 capture cannot "
                "see native windows; install a Wayland screenshot tool such as `spectacle`."
            )

        image = Image.frombytes("RGB", shot.size, shot.rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _capture(self, monitor: int) -> str:
        if self._on_wayland():
            data = self._capture_wayland()
            if data:
                return data
        return self._capture_x11(monitor)


if __name__ == "__main__":
    ScreenVision().run()
