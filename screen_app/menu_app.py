# /home/lubuharg/Documents/MTG/screen_app/menu_app.py
import os
import subprocess
import sys
from pathlib import Path
import json   # <-- add this
import cv2
import threading
import time
import numpy as np
from picamera2 import Picamera2
from libcamera import controls   # <-- add this

# Kivy env
os.environ["KIVY_NO_ARGS"] = "1"
os.environ["KIVY_WINDOW"] = "sdl2"
os.environ["KIVY_INPUT"] = "mouse,multitouch_on_demand"

from kivy.config import Config

Config.set("input", "mouse", "mouse,multitouch_on_demand")
Config.set("input", "mtdev_%(name)s", "")
Config.set("input", "hid_%(name)s", "")
Config.set("graphics", "fullscreen", "auto")
Config.set("kivy", "exit_on_escape", "0")

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.scrollview import ScrollView
from kivy.uix.checkbox import CheckBox
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.graphics.texture import Texture

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # /home/.../Documents/MTG
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "card_detector.pt"
START_SCRIPT = PROJECT_ROOT / "start.py"

# Prefer a known venv unless the user already set an override.
_DEFAULT_PYTHON = Path.home() / "labelimg-venv" / "bin" / "python"
if not os.environ.get("MTG_PYTHON") and _DEFAULT_PYTHON.exists():
    os.environ["MTG_PYTHON"] = str(_DEFAULT_PYTHON)

def _maybe_add_venv_site_packages():
    """
    If the app was launched outside a virtualenv, try to add a venv's site-packages
    to sys.path so imports like ultralytics can be resolved. Preference order:
    1) MTG_VENV env var, 2) VIRTUAL_ENV, 3) .venv under project root,
    4) ~/labelimg-venv (current known env on device).
    """
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = []

    for env_var in ("MTG_VENV", "VIRTUAL_ENV"):
        env_val = os.environ.get(env_var)
        if env_val:
            candidates.append(Path(env_val))

    candidates.append(PROJECT_ROOT / ".venv")
    candidates.append(Path.home() / "labelimg-venv")

    for root in candidates:
        site_dir = root / "lib" / py_ver / "site-packages"
        if site_dir.is_dir() and str(site_dir) not in sys.path:
            sys.path.insert(0, str(site_dir))
            print(f"[DebugScreen] Added venv site-packages: {site_dir}")
            return site_dir

    return None


ADDED_VENV_SITE = _maybe_add_venv_site_packages()

# YOLO import is optional; keep UI usable even if the dependency is missing.
try:
    from detect_card_yolo import get_model  # type: ignore
    YOLO_AVAILABLE = True
except Exception as e:
    print("[DebugScreen] YOLO import failed:", repr(e))
    get_model = None
    YOLO_AVAILABLE = False


# --------- CONSTANTS / PATHS ---------
SPLASH_IMAGE = "/home/lubuharg/Documents/MTG/Documents/Forest Finder Logo.jpg"
FOCUS_FILE = "/home/lubuharg/Documents/MTG/config/focus.json"

# Base dir = folder where THIS script lives: /home/.../MTG/screen_app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# tests folder is one level up: /home/.../MTG/tests
TESTS_DIR = os.path.join(BASE_DIR, "..", "tests")

TEST_SORTER = os.path.join(TESTS_DIR, "test_servoMotor360.py")      # Sorter
TEST_DROPPER = os.path.join(TESTS_DIR, "test_servoMotor180.py")     # Card Dropper
TEST_PULLER = os.path.join(TESTS_DIR, "test_servoMotor360v2.py")    # Card Puller



# --------- WIFI HELPERS (NetworkManager / nmcli) ---------
def get_current_ssid():
    """
    Return current WiFi SSID or None, using nmcli.
    """
    try:
        out = subprocess.check_output(
            ["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        for line in out.splitlines():
            active, ssid = (line.split(":", 1) + [""])[:2]
            if active == "yes" and ssid:
                return ssid
        return None
    except Exception:
        return None


def scan_wifi_networks():
    """
    Return a list of (ssid, signal) sorted by signal desc.
    """
    try:
        out = subprocess.check_output(
            ["nmcli", "-t", "-f", "SSID,SIGNAL", "dev", "wifi"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        ssid_map = {}
        for line in out.splitlines():
            ssid, signal = (line.split(":", 1) + [""])[:2]
            ssid = ssid.strip()
            if not ssid:
                continue
            try:
                sig = int(signal)
            except Exception:
                sig = 0
            if ssid not in ssid_map or sig > ssid_map[ssid]:
                ssid_map[ssid] = sig

        return sorted(ssid_map.items(), key=lambda kv: kv[1], reverse=True)
    except Exception:
        return []


def connect_wifi(ssid, password, remember=True):
    """
    Connect to WiFi using nmcli.
    If remember=False, try to disable autoconnect on that connection.
    """
    try:
        cmd = ["nmcli", "device", "wifi", "connect", ssid, "password", password]
        subprocess.check_call(cmd)

        # NetworkManager remembers connections by default.
        # Use autoconnect yes/no for the checkbox behaviour:
        try:
            auto_val = "yes" if remember else "no"
            subprocess.check_call(
                ["nmcli", "connection", "modify", ssid, "connection.autoconnect", auto_val],
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

        return True, f"Connected to {ssid}"
    except FileNotFoundError:
        return False, "nmcli not found. Configure WiFi manually or install NetworkManager."
    except subprocess.CalledProcessError as e:
        return False, f"Failed to connect: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


# ---------------- Splash Screen ----------------
class SplashScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(name="splash", **kwargs)

        root = BoxLayout(orientation="vertical", padding=16, spacing=12)
        img = Image(
            source=SPLASH_IMAGE,
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1, 0.82),
        )
        root.add_widget(img)

        start_btn = Button(text="Start", font_size="24sp", size_hint=(1, 0.18))
        start_btn.bind(on_press=lambda *_: self.goto_menu())
        root.add_widget(start_btn)
        self.add_widget(root)

    def goto_menu(self):
        btn = self.children[0].children[0]  # the Start button
        Animation(d=0.08, opacity=0.7).start(btn)
        Clock.schedule_once(lambda *_: setattr(btn, "opacity", 1), 0.12)
        self.manager.current = "menu"


# ---------------- Main Menu Screen ----------------
class Menu(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", spacing=16, padding=16, **kwargs)
        self.scan_thread = None
        self._scan_proc = None
        self._scan_cancelled = False
        self._preview_popup = None
        self._scan_popup_mode = None
        self._scan_preview_img = None

        self.add_widget(
            Label(
                text="[b]Forest-Finder[/b]",
                markup=True,
                font_size="28sp",
                size_hint=(1, 0.2),
            )
        )
        self.add_widget(self._btn("Start Scan", self.on_start_scan))
        self.add_widget(self._btn("Show Last Result", self.on_show_last))
        self.add_widget(self._btn("Settings", self.on_settings))
        self.add_widget(self._btn("Shutdown", self.on_shutdown))
        self.status = Label(text="", size_hint=(1, 0.2))
        self.add_widget(self.status)

    def _btn(self, text, cb):
        return Button(text=text, font_size="22sp", size_hint=(1, 0.15), on_press=cb)

    def on_start_scan(self, *_):
        # avoid concurrent scans
        if self.scan_thread and self.scan_thread.is_alive():
            self._flash("Scan already running...")
            return
        self._scan_cancelled = False
        self._show_scan_popup("Starting scan...")
        self.scan_thread = threading.Thread(target=self._run_scan, daemon=True)
        self.scan_thread.start()

    def on_show_last(self, *_):
        self._flash("Opening last result...")

    def on_settings(self, *_):
        from kivy.app import App

        self._flash("Opening settings...")
        Clock.schedule_once(
            lambda *_: setattr(App.get_running_app().root, "current", "settings"),
            0.1,
        )

    def on_shutdown(self, *_):
        self._flash("Shutting down...")
        Clock.schedule_once(lambda *_: os.system("sudo shutdown -h now"), 1.5)

    def _flash(self, msg):
        self.status.text = msg
        Clock.schedule_once(lambda *_: self._clear(), 2)

    def _clear(self):
        self.status.text = ""

    # --- Scan helpers ---
    def _show_scan_popup(self, message: str):
        box = BoxLayout(orientation="vertical", padding=20, spacing=10)
        label = Label(text=message, font_size="20sp", halign="center", valign="middle")
        label.bind(size=lambda inst, *_: setattr(inst, "text_size", inst.size))
        box.add_widget(label)

        cancel_btn = Button(
            text="Cancel scanning",
            size_hint=(1, 0.35),
            font_size="18sp",
        )
        cancel_btn.bind(on_press=self._cancel_scan)
        box.add_widget(cancel_btn)

        self._scan_popup_label = label
        self._scan_popup_cancel_btn = cancel_btn
        self._scan_popup = Popup(
            title="Scanning",
            content=box,
            size_hint=(0.6, 0.3),
            auto_dismiss=False,
        )
        self._scan_popup_mode = "scan"
        self._scan_popup.open()

    def _update_scan_popup(self, message: str):
        label = getattr(self, "_scan_popup_label", None)
        if label is not None:
            label.text = message

    def _cancel_scan(self, *_):
        # Disable the button to avoid repeated clicks.
        btn = getattr(self, "_scan_popup_cancel_btn", None)
        if btn is not None:
            btn.disabled = True
        self._scan_cancelled = True
        self._update_scan_popup("Cancelling scan...")
        proc = getattr(self, "_scan_proc", None)
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                print("[Menu] Sent terminate to scanning process.")
            except Exception as e:
                print("[Menu] Failed to terminate scan process:", e)

    def _dismiss_scan_popup(self):
        popup = getattr(self, "_scan_popup", None)
        if popup is not None:
            popup.dismiss()
            self._scan_popup = None
            self._scan_popup_label = None
            self._scan_popup_cancel_btn = None
            self._scan_popup_mode = None
            self._scan_preview_img = None

    def _python_cmd(self) -> str:
        """
        Choose a Python interpreter for running start.py.
        Preference: MTG_PYTHON env, project .venv, ~/labelimg-venv, current interpreter.
        """
        candidates = []
        env_val = os.environ.get("MTG_PYTHON")
        if env_val:
            candidates.append(Path(env_val))
        candidates.append(PROJECT_ROOT / ".venv" / "bin" / "python")
        candidates.append(Path.home() / "labelimg-venv" / "bin" / "python")
        candidates.append(Path(sys.executable))
        for c in candidates:
            if c and Path(c).exists():
                print(f"[Menu] Using Python interpreter: {c}")
                return str(c)
        print("[Menu] Falling back to current interpreter for scan.")
        return sys.executable

    def _run_scan(self):
        cmd = [self._python_cmd(), str(START_SCRIPT)]
        Clock.schedule_once(lambda *_: self._update_scan_popup("Running scan..."), 0)
        proc = None
        last_line = ""
        ret = -1

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._scan_proc = proc
            if proc.stdout:
                for raw in proc.stdout:
                    line = raw.strip()
                    if not line:
                        continue
                    last_line = line
                    if line.startswith("[PREVIEW]"):
                        payload = line[len("[PREVIEW]"):].strip()
                        if not payload:
                            continue
                        parts = payload.split()
                        path = parts[0]
                        note = " ".join(parts[1:]).strip()
                        Clock.schedule_once(lambda _dt, _p=path, _n=note: self._show_preview_popup(_p, note=_n), 0)
                    elif line.startswith("[PIPELINE]"):
                        # Strip the [PIPELINE] tag and shorten noisy messages (paths, etc.)
                        status = line[len("[PIPELINE]"):].strip()

                        # If it contains what looks like a path, trim it to a nice short label
                        if " /home/" in status:
                            # e.g. "Captured /home/..." -> "Captured image"
                            if status.lower().startswith("captured"):
                                status = "Captured image"
                            else:
                                status = status.split(" /home", 1)[0].strip()

                        # Fallback: don't ever show more than 120 chars
                        short_status = status[:120] if status else "Working..."

                        Clock.schedule_once(
                            lambda _dt, _m=short_status: self._update_scan_popup(_m),
                            0,
                        )
            ret = proc.wait()
        except Exception as e:
            last_line = f"Scan error: {e}"
            ret = -1
        finally:
            self._scan_proc = None
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass

        if self._scan_cancelled:
            msg = "Scan cancelled."
        elif ret == 0:
            msg = "Scan done."
            if last_line:
                msg += f" {last_line[:120]}"
        else:
            msg = f"Scan failed (code {ret})."
            if last_line:
                msg += f" {last_line[:120]}"

        def _finish(*_):
            self._dismiss_scan_popup()
            self._flash(msg)
            self.scan_thread = None

        Clock.schedule_once(_finish, 0)

    def _show_preview_popup(self, img_path: str, note: str = ""):
        """
        Show the warped/ocr image for a few seconds when the pipeline
        reports a new preview path.
        """
        try:
            path = Path(img_path)
        except Exception:
            return

        if not path.exists():
            print(f"[Menu] Preview image not found: {img_path}")
            return

        # If we already have a preview popup, reuse it by swapping the image only.
        if getattr(self, "_scan_popup_mode", None) == "preview" and getattr(self, "_scan_popup", None):
            try:
                if getattr(self, "_scan_preview_img", None):
                    self._scan_preview_img.source = str(path)
                    self._scan_preview_img.reload()
                return
            except Exception:
                # Fall through to recreating the popup.
                pass

        # Otherwise close any lingering scanning popup before showing the preview.
        self._dismiss_scan_popup()

        box = BoxLayout(orientation="vertical", padding=10, spacing=10)
        img = Image(
            source=str(path),
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1, 0.85),   # was 0.7 – give more space to the card
        )
        box.add_widget(img)

        cancel_btn = Button(
            text="Cancel",
            size_hint=(1, 0.15),
            font_size="18sp",
        )
        cancel_btn.bind(on_press=self._cancel_scan)
        box.add_widget(cancel_btn)

        popup = Popup(
            title="Detected card",
            content=box,
            size_hint=(0.9, 0.9),
            auto_dismiss=False,
        )

        self._scan_popup = popup
        self._scan_popup_cancel_btn = cancel_btn
        self._scan_popup_mode = "preview"
        self._scan_preview_img = img
        popup.open()


class MenuScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(name="menu", **kwargs)
        self.container = Menu()
        self.container.opacity = 0
        self.add_widget(self.container)
        self.bind(on_enter=self._fade_in)

    def _fade_in(self, *_):
        Animation(opacity=1, d=0.25, t="out_quad").start(self.container)


# ---------------- Settings Screen ----------------
class SettingsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(name="settings", **kwargs)

        root = BoxLayout(orientation="vertical", padding=16, spacing=16)

        root.add_widget(
            Label(
                text="[b]Settings[/b]",
                markup=True,
                font_size="26sp",
                size_hint=(1, 0.2),
            )
        )

        btn_wlan = Button(text="WLAN Settings", font_size="22sp", size_hint=(1, 0.15))
        btn_wlan.bind(on_press=lambda *_: setattr(self.manager, "current", "wlan"))
        root.add_widget(btn_wlan)

        btn_debug = Button(text="Debug", font_size="22sp", size_hint=(1, 0.15))
        btn_debug.bind(on_press=lambda *_: setattr(self.manager, "current", "debug"))
        root.add_widget(btn_debug)

        # NEW: Install submenu
        btn_install = Button(text="Install", font_size="22sp", size_hint=(1, 0.15))
        btn_install.bind(on_press=lambda *_: setattr(self.manager, "current", "install"))
        root.add_widget(btn_install)

        back_btn = Button(text="Back", font_size="20sp", size_hint=(1, 0.15))

        back_btn.bind(on_press=lambda *_: setattr(self.manager, "current", "menu"))
        root.add_widget(back_btn)

        self.add_widget(root)


# ---------------- Simple On-screen Keyboard ----------------
class SimpleKeyboard(Popup):
    """
    Simple on-screen keyboard with:
    - digits 0–9
    - lowercase letters
    - Shift toggle for uppercase
    - some symbols: - _ @ . ! ?
    Password is shown as plain text.
    """

    def __init__(self, target_input, **kwargs):
        super().__init__(
            title="Enter Password",
            size_hint=(0.9, 0.6),
            auto_dismiss=False,
            **kwargs,
        )
        self.target_input = target_input
        self.upper = False  # Shift state

        main = BoxLayout(orientation="vertical", spacing=5, padding=5)

        # Display current text – NOT masked
        self.display = TextInput(
            text=target_input.text,
            multiline=False,
            password=False,
            size_hint=(1, 0.25),
        )
        main.add_widget(self.display)

        # Keys grid
        keys_layout = GridLayout(cols=10, size_hint=(1, 0.55), spacing=2)

        rows = [
            "1234567890",
            "qwertyuiop",
            "asdfghjkl",
            "zxcvbnm",
            "-_@.!?",
        ]

        for row in rows:
            for ch in row:
                btn = Button(text=ch)
                btn.bind(on_press=lambda inst, c=ch: self._add_char(c))
                keys_layout.add_widget(btn)

        main.add_widget(keys_layout)

        # Bottom row: Shift, space, backspace, clear, OK, cancel
        bottom = GridLayout(cols=6, size_hint=(1, 0.2), spacing=2)

        self.shift_btn = Button(text="Shift")
        self.shift_btn.bind(on_press=self._toggle_shift)
        bottom.add_widget(self.shift_btn)

        space_btn = Button(text="Space")
        space_btn.bind(on_press=lambda *_: self._add_char(" "))
        bottom.add_widget(space_btn)

        back_btn = Button(text="Back")
        back_btn.bind(on_press=lambda *_: self._backspace())
        bottom.add_widget(back_btn)

        clear_btn = Button(text="Clear")
        clear_btn.bind(on_press=lambda *_: self._clear())
        bottom.add_widget(clear_btn)

        ok_btn = Button(text="OK")
        ok_btn.bind(on_press=lambda *_: self._ok())
        bottom.add_widget(ok_btn)

        cancel_btn = Button(text="Cancel")
        cancel_btn.bind(on_press=lambda *_: self.dismiss())
        bottom.add_widget(cancel_btn)

        main.add_widget(bottom)
        self.content = main

    def _toggle_shift(self, *_):
        self.upper = not self.upper
        self.shift_btn.text = "SHIFT" if self.upper else "Shift"

    def _add_char(self, c):
        if self.upper and c.isalpha():
            c = c.upper()
        self.display.text += c

    def _backspace(self):
        self.display.text = self.display.text[:-1]

    def _clear(self):
        self.display.text = ""

    def _ok(self):
        self.target_input.text = self.display.text
        self.dismiss()


# ---------------- WLAN Settings Screen ----------------
class WlanSettingsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(name="wlan", **kwargs)

        self.layout = BoxLayout(orientation="vertical", padding=16, spacing=16)

        self.layout.add_widget(
            Label(
                text="[b]WLAN Settings[/b]",
                markup=True,
                font_size="26sp",
                size_hint=(1, 0.12),
            )
        )

        # Current WiFi
        self.current_label = Label(
            text="Current WiFi: [b]-[/b]",
            markup=True,
            font_size="18sp",
            size_hint=(1, 0.08),
        )
        self.layout.add_widget(self.current_label)

        # Scan button
        scan_btn = Button(text="Scan Networks", size_hint=(1, 0.1))
        scan_btn.bind(on_press=self.refresh_networks)
        self.layout.add_widget(scan_btn)

        # Scrollable list of networks
        self.networks_box = BoxLayout(
            orientation="vertical", size_hint_y=None, spacing=4
        )
        self.networks_box.bind(minimum_height=self.networks_box.setter("height"))

        scroll = ScrollView(size_hint=(1, 0.3))
        scroll.add_widget(self.networks_box)
        self.layout.add_widget(scroll)

        # Selected SSID
        self.layout.add_widget(Label(text="Selected SSID:", size_hint=(1, 0.06)))
        self.ssid_input = TextInput(multiline=False, readonly=True, size_hint=(1, 0.08))
        self.layout.add_widget(self.ssid_input)

        # Password + keyboard button
        pw_row = BoxLayout(orientation="horizontal", size_hint=(1, 0.1), spacing=8)
        pw_row.add_widget(Label(text="Password:", size_hint=(0.4, 1)))
        self.password_input = TextInput(
            multiline=False,
            password=False,  # visible
            size_hint=(0.4, 1),
        )
        pw_row.add_widget(self.password_input)

        kb_btn = Button(text="Keyboard", size_hint=(0.2, 1))
        kb_btn.bind(on_press=self.open_keyboard)
        pw_row.add_widget(kb_btn)

        self.layout.add_widget(pw_row)

        # Remember checkbox
        rem_row = BoxLayout(orientation="horizontal", size_hint=(1, 0.08), spacing=8)
        rem_row.add_widget(
            Label(text="Remember this WiFi (autoconnect)", size_hint=(0.8, 1))
        )
        self.remember_cb = CheckBox(size_hint=(0.2, 1), active=True)
        rem_row.add_widget(self.remember_cb)
        self.layout.add_widget(rem_row)

        # Status label
        self.status = Label(text="", markup=True, size_hint=(1, 0.1))
        self.layout.add_widget(self.status)

        # Connect + Back buttons
        btn_row = BoxLayout(orientation="horizontal", size_hint=(1, 0.12), spacing=8)

        connect_btn = Button(text="Connect", size_hint=(0.6, 1))
        connect_btn.bind(on_press=self.on_connect)
        btn_row.add_widget(connect_btn)

        back_btn = Button(text="Back", size_hint=(0.4, 1))
        back_btn.bind(
            on_press=lambda *_: setattr(self.manager, "current", "settings")
        )
        btn_row.add_widget(back_btn)

        self.layout.add_widget(btn_row)

        self.add_widget(self.layout)

        self.refresh_ssid()
        self.refresh_networks()

    def refresh_ssid(self, *_):
        ssid = get_current_ssid()
        if ssid:
            self.current_label.text = f"Current WiFi: [b]{ssid}[/b]"
        else:
            self.current_label.text = "Current WiFi: [b]Not connected[/b]"

    def refresh_networks(self, *_):
        self.networks_box.clear_widgets()
        networks = scan_wifi_networks()
        if not networks:
            self.networks_box.add_widget(
                Label(
                    text="No networks found",
                    size_hint_y=None,
                    height=40,
                )
            )
            return

        for ssid, signal in networks:
            btn = Button(
                text=f"{ssid}  ({signal}%)",
                size_hint_y=None,
                height=40,
            )
            btn.bind(on_press=lambda inst, s=ssid: self.select_ssid(s))
            self.networks_box.add_widget(btn)

    def select_ssid(self, ssid):
        self.ssid_input.text = ssid
        self.password_input.text = ""
        self.status.text = ""

    def open_keyboard(self, *_):
        kb = SimpleKeyboard(self.password_input)
        kb.open()

    def on_connect(self, *_):
        ssid = self.ssid_input.text.strip()
        password = self.password_input.text.strip()
        if not ssid:
            self.status.text = "[color=ff3333]Please select a WiFi[/color]"
            return
        if not password:
            self.status.text = "[color=ff3333]Password required[/color]"
            return

        remember = self.remember_cb.active
        self.status.text = "Connecting..."
        Clock.schedule_once(
            lambda *_: self._do_connect(ssid, password, remember),
            0,
        )

    def _do_connect(self, ssid, password, remember):
        ok, msg = connect_wifi(ssid, password, remember)
        color = "33ff33" if ok else "ff3333"
        self.status.text = f"[color={color}]{msg}[/color]"
        self.refresh_ssid()


class InstallScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(name="install", **kwargs)

        root = BoxLayout(orientation="vertical", padding=16, spacing=16)

        root.add_widget(
            Label(
                text="[b]Network Repair[/b]",
                markup=True,
                font_size="26sp",
                size_hint=(1, 0.2),
            )
        )

        self.status = Label(text="", markup=True, size_hint=(1, 0.2))
        root.add_widget(self.status)

        btn_fix = Button(
            text="Repair Network",
            font_size="22sp",
            size_hint=(1, 0.2),
        )
        btn_fix.bind(on_press=self.on_repair)
        root.add_widget(btn_fix)

        back_btn = Button(
            text="Back",
            font_size="20sp",
            size_hint=(1, 0.2),
        )
        back_btn.bind(on_press=lambda *_: setattr(self.manager, "current", "settings"))
        root.add_widget(back_btn)

        self.add_widget(root)

    def on_repair(self, *_):
        self.status.text = "[color=ffff33]Repairing network...[/color]"
        t = threading.Thread(target=self._repair_worker, daemon=True)
        t.start()

    def _repair_worker(self):
        cmds = [
            ["sudo", "nmcli", "radio", "wifi", "on"],
            ["sudo", "nmcli", "device", "disconnect", "wlan0"],
            ["sudo", "nmcli", "device", "connect", "wlan0"],
            ["sudo", "systemctl", "restart", "NetworkManager"],
        ]

        ok = True
        err = ""

        for c in cmds:
            try:
                print("[Repair] Running:", " ".join(c))
                subprocess.check_call(c)
            except Exception as e:
                ok = False
                err = f"{' '.join(c)} failed: {e}"
                print("[Repair] ERROR:", err)
                break

        # Collect final status
        try:
            out = subprocess.check_output(["nmcli", "-t", "-f", "DEVICE,STATE", "device"]).decode()
        except:
            out = "Could not read status."

        msg = (
            f"[color=33ff33]Network repaired successfully![/color]\n{out}"
            if ok else
            f"[color=ff3333]Error during repair: {err}[/color]"
        )

        Clock.schedule_once(lambda *_: setattr(self.status, "text", msg), 0)



# ---------------- Debug Screen ----------------

class DebugScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(name="debug", **kwargs)

        root = BoxLayout(orientation="vertical", padding=16, spacing=16)

        root.add_widget(
            Label(
                text="[b]Debug[/b]",
                markup=True,
                font_size="26sp",
                size_hint=(1, 0.2),
            )
        )

        btn_sorter = Button(
            text="Test Card Puller (360° v2)",
            font_size="20sp",
            size_hint=(1, 0.15),
        )
        btn_sorter.bind(on_press=lambda *_: self.run_test("Sorter", TEST_SORTER))
        root.add_widget(btn_sorter)

        btn_dropper = Button(
            text="Test Card Dropper (180°)",
            font_size="20sp",
            size_hint=(1, 0.15),
        )
        btn_dropper.bind(on_press=lambda *_: self.run_test("Card Dropper", TEST_DROPPER))
        root.add_widget(btn_dropper)

        btn_puller = Button(
            text="Test Sorter (360°)",
            font_size="20sp",
            size_hint=(1, 0.15),
        )
        btn_puller.bind(on_press=lambda *_: self.run_test("Card Puller", TEST_PULLER))
        root.add_widget(btn_puller)

        # NEW BUTTON for camera + detection
        btn_camera = Button(
            text="Camera test + card detection",
            font_size="20sp",
            size_hint=(1, 0.15),
        )
        btn_camera.bind(on_press=lambda *_: self.run_camera_test_with_detection())
        root.add_widget(btn_camera)

        yolo_text = (
            f"[color=33ff33]YOLO ready ({YOLO_MODEL_PATH.name})[/color]"
            if YOLO_AVAILABLE and YOLO_MODEL_PATH.exists()
            else "[color=ffcc00]YOLO missing – install ultralytics + place model at models/card_detector.pt[/color]"
        )
        self.yolo_status = Label(text=yolo_text, markup=True, size_hint=(1, 0.1))
        root.add_widget(self.yolo_status)

        self.status = Label(text="", markup=True, size_hint=(1, 0.2))
        root.add_widget(self.status)

        back_btn = Button(
            text="Back",
            font_size="20sp",
            size_hint=(1, 0.15),
        )
        back_btn.bind(
            on_press=lambda *_: setattr(self.manager, "current", "settings")
        )
        root.add_widget(back_btn)

        self.add_widget(root)

    # --- popup + background thread + full-screen preview ---

    def run_camera_test_with_detection(self):
        """Show 'loading...' popup and start camera+detect in background."""
        box = BoxLayout(orientation="vertical", padding=20)
        label = Label(text="Loading...\nPlease wait.", font_size="20sp")
        box.add_widget(label)

        self._loading_popup = Popup(
            title="Camera Test",
            content=box,
            size_hint=(0.6, 0.3),
            auto_dismiss=False,
        )

        self._loading_popup.open()
        self.status.text = "[color=ffff33]Starting camera test...[/color]"

        # Run heavy work in a separate thread (no UI calls inside!)
        t = threading.Thread(target=self._camera_test_worker, daemon=True)
        t.start()

    def _camera_test_worker(self):
        """Background thread: capture image, run detect + YOLO, prepare RGB image."""
        try:
            print("[DebugScreen] Capturing frame in worker...")

            # --- Load calibrated lens position ---
            lens_pos = None
            try:
                with open(FOCUS_FILE) as f:
                    cfg_focus = json.load(f)
                lens_pos = cfg_focus.get("LensPosition", None)
                print(f"[DebugScreen] Loaded LensPosition from focus.json: {lens_pos}")
            except Exception as e_focus:
                print("[DebugScreen] Could not read focus file:", e_focus)

            picam2 = Picamera2()

            # Use a 4:3 frame so the portrait preview has more width and less visible crop
            config = picam2.create_still_configuration({"size": (1640, 1232)})
            picam2.configure(config)
            picam2.start()
            time.sleep(0.2)  # small warm-up like in your test script

            # --- Apply manual focus using stored LensPosition ---
            if lens_pos is not None:
                try:
                    picam2.set_controls({
                        "AfMode": controls.AfModeEnum.Manual,
                        "LensPosition": float(lens_pos),
                    })
                    print("[DebugScreen] Manual focus set via controls.AfModeEnum.Manual")
                except Exception as e_manual:
                    # Fallback numeric AfMode as in your script
                    print("[DebugScreen] Manual AF via enum failed, fallback to numeric:", e_manual)
                    try:
                        picam2.set_controls({
                            "AfMode": 0,  # many builds: 0 = Manual
                            "LensPosition": float(lens_pos),
                        })
                        print("[DebugScreen] Manual focus set via AfMode=0")
                    except Exception as e_fallback:
                        print("[DebugScreen] Could not set manual focus:", e_fallback)
            else:
                print("[DebugScreen] WARNING: No lens_pos loaded, using default focus.")

            # Snap frame quickly (no autofocus delay)
            frame = picam2.capture_array()  # RGB
            picam2.stop()
            picam2.close()

            print("[DebugScreen] Raw frame shape:", frame.shape)

            # Optional: save raw frame for debugging
            try:
                out_dir = "/home/lubuharg/Documents/MTG/tests/output"
                os.makedirs(out_dir, exist_ok=True)
                raw_path = os.path.join(out_dir, "camera_live_raw.jpg")
                cv2.imwrite(raw_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"[DebugScreen] Saved raw frame to {raw_path}")
            except Exception as e_save:
                print("[DebugScreen] Could not save debug image:", e_save)

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Keep an upright version only for fallback preview; YOLO runs on the original.
            frame_bgr_upright = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

            # --- YOLO detection for cropped card preview (no contour detect) ---
            crop_rgb = None
            if YOLO_AVAILABLE:
                try:
                    crop_rgb, yolo_status = self._apply_yolo(frame_bgr)
                    msg = yolo_status
                except Exception as e_yolo:
                    print("[DebugScreen] YOLO error:", repr(e_yolo))
                    msg = f"YOLO error: {e_yolo}"
            else:
                msg = "YOLO unavailable"

            # Use the cropped card for the preview if we have it; otherwise show the full frame.
            if crop_rgb is not None:
                # Rotate to match the upright preview orientation.
                crop_rgb = cv2.rotate(crop_rgb, cv2.ROTATE_90_CLOCKWISE)
                img_rgb = crop_rgb
            else:
                img_rgb = cv2.cvtColor(frame_bgr_upright, cv2.COLOR_BGR2RGB)

            h, w, _ = img_rgb.shape
            print(f"[DebugScreen] Image for preview: {w}x{h}")

            # downscale if too large (avoid texture issues)
            max_side = 1280
            if max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                new_w = int(w * scale)
                new_h = int(h * scale)
                print(f"[DebugScreen] Resizing for preview to {new_w}x{new_h}")
                img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # schedule UI update on main thread
            Clock.schedule_once(
                lambda dt, img=img_rgb, m=msg, crop=crop_rgb: self._camera_test_done(img, m, crop)
            )
        except Exception as e:
            print("[DebugScreen] Error in worker:", repr(e))
            Clock.schedule_once(lambda dt, err=e: self._camera_test_error(err))

    def _apply_yolo(self, frame_bgr):
        """Run YOLO on the raw frame and return the cropped card (RGB) plus status."""
        if not YOLO_AVAILABLE or get_model is None:
            return None, "YOLO unavailable"
        model = get_model()
        results = model(frame_bgr, conf=0.02)[0]
        if not results.boxes:
            print("[DebugScreen][YOLO] No detections.")
            return None, "YOLO: no card detected"

        boxes = results.boxes
        best_idx = boxes.conf.argmax().item()
        box = boxes[best_idx]
        best_conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        H, W, _ = frame_bgr.shape
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))
        if x2 <= x1 or y2 <= y1:
            print("[DebugScreen][YOLO] Invalid crop coords:", (x1, y1, x2, y2))
            return None, "YOLO: bad box"

        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        print(f"[DebugScreen][YOLO] Crop size: {crop_rgb.shape[1]}x{crop_rgb.shape[0]}, conf={best_conf:.3f}")
        return crop_rgb, f"YOLO: card conf={best_conf:.2f}"

    def _camera_test_done(self, img_rgb, msg: str, crop_rgb=None):
        """Runs on UI thread after worker finishes."""
        # Close popup
        if hasattr(self, "_loading_popup") and self._loading_popup:
            self._loading_popup.dismiss()
            self._loading_popup = None

        # Update status on Debug screen (for when user comes back)
        if crop_rgb is None:
            self.status.text = "[color=ffcc00]No card detected.[/color]"
        else:
            self.status.text = "[color=33ff33]Card detected.[/color]"

        # Send image to CameraPreviewScreen and switch there
        if self.manager:
            screen = self.manager.get_screen("camera_preview")
            screen.set_image_from_numpy(img_rgb, crop_rgb=crop_rgb, msg=msg)
            self.manager.current = "camera_preview"

    def _camera_test_error(self, e: Exception):
        """UI-thread error handler."""
        if hasattr(self, "_loading_popup") and self._loading_popup:
            self._loading_popup.dismiss()
            self._loading_popup = None

        self.status.text = f"[color=ff3333]Error: {e}[/color]"
        print("[DebugScreen] Camera test error:", repr(e))

    def run_test(self, name, script_path):
        if not os.path.exists(script_path):
            self.status.text = f"[color=ff3333]Script not found: {script_path}[/color]"
            return

        self.status.text = f"Starting {name} test..."
        try:
            subprocess.Popen([sys.executable, script_path])
            self.status.text = f"[color=33ff33]{name} test started.[/color]"
        except Exception as e:
            self.status.text = f"[color=ff3333]Error: {e}[/color]"


class CameraPreviewScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(name="camera_preview", **kwargs)

        root = BoxLayout(orientation="vertical", padding=10, spacing=10)

        # Keep the status label compact so the preview can dominate the screen.
        self.status_label = Label(
            text="",
            markup=True,
            size_hint=(1, None),
            height=40,
        )
        root.add_widget(self.status_label)

        # Main preview area: show only the final image.
        body = BoxLayout(orientation="vertical", size_hint=(1, 0.85), spacing=10)

        self.image = Image(
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1, 1),
        )
        body.add_widget(self.image)
        root.add_widget(body)

        # Bottom bar with back button
        bottom = BoxLayout(
            orientation="horizontal",
            size_hint=(1, 0.1),
            padding=10,
            spacing=10,
        )

        back_btn = Button(
            text="Back to Debug",
            font_size="18sp",
            size_hint=(0.3, 1),
        )
        back_btn.bind(on_press=self._go_back)

        bottom.add_widget(back_btn)
        root.add_widget(bottom)

        self.add_widget(root)

    def _go_back(self, *_):
        if self.manager:
            self.manager.current = "debug"

    def _apply_texture(self, widget, img_rgb):
        h, w, _ = img_rgb.shape
        texture = Texture.create(size=(w, h), colorfmt="rgb")
        texture.blit_buffer(img_rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        texture.flip_vertical()
        widget.texture = texture
        widget.texture_size = texture.size
        widget.canvas.ask_update()

    def set_image_from_numpy(self, img_rgb, *, crop_rgb=None, msg: str = ""):
        """Receive an RGB numpy image and show only the final result."""
        display = crop_rgb if crop_rgb is not None else img_rgb
        h, w, _ = display.shape
        suffix = " (cropped)" if crop_rgb is not None else ""
        print(f"[CameraPreviewScreen] Showing image {w}x{h}{suffix}")
        self._apply_texture(self.image, display)

        self.status_label.text = msg


# ---------------- App ----------------
class FFMenuApp(App):
    def build(self):
        self.title = "Forest-Finder"
        Window.multitouch_on_demand = True

        def _hide_cursor(*_):
            try:
                Window.show_cursor = False
            except Exception:
                pass

        Window.bind(on_draw=_hide_cursor)

        sm = ScreenManager(transition=FadeTransition(duration=0.28))
        sm.add_widget(SplashScreen())
        sm.add_widget(MenuScreen())
        sm.add_widget(SettingsScreen())
        sm.add_widget(WlanSettingsScreen())
        sm.add_widget(InstallScreen())    
        sm.add_widget(DebugScreen())
        sm.add_widget(CameraPreviewScreen())
        sm.current = "splash"
        return sm


if __name__ == "__main__":
    FFMenuApp().run()
