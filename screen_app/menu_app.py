# /home/lubuharg/Documents/MTG/screen_app/menu_app.py
import os
os.environ['KIVY_NO_ARGS'] = '1'
os.environ['KIVY_WINDOW'] = 'sdl2'
os.environ['KIVY_INPUT'] = 'mouse,multitouch_on_demand'

from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('input', 'mtdev_%(name)s', '')
Config.set('input', 'hid_%(name)s', '')
Config.set('graphics', 'fullscreen', 'auto')
Config.set('kivy', 'exit_on_escape', '0')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition  # CHANGED
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.animation import Animation  # NEW

SPLASH_IMAGE = "/home/lubuharg/Documents/MTG/Documents/Forest Finder Logo.jpg"

# ---------------- Splash Screen ----------------
class SplashScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(name="splash", **kwargs)

        root = BoxLayout(orientation="vertical", padding=16, spacing=12)
        img = Image(source=SPLASH_IMAGE, allow_stretch=True, keep_ratio=True, size_hint=(1, 0.82))
        root.add_widget(img)

        start_btn = Button(text="Start", font_size="24sp", size_hint=(1, 0.18))
        start_btn.bind(on_press=lambda *_: self.goto_menu())
        root.add_widget(start_btn)
        self.add_widget(root)

    def goto_menu(self):
        # optional tiny click feedback
        btn = self.children[0].children[0]  # the Start button
        Animation(d=0.08, opacity=0.7).start(btn)
        Clock.schedule_once(lambda *_: setattr(btn, "opacity", 1), 0.12)
        self.manager.current = "menu"  # fade handled by ScreenManager

# ---------------- Your Menu Screen ----------------
class Menu(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", spacing=16, padding=16, **kwargs)

        self.add_widget(Label(text="[b]Forest-Finder[/b]", markup=True, font_size="28sp", size_hint=(1, 0.2)))
        self.add_widget(self._btn("Start Scan", self.on_start_scan))
        self.add_widget(self._btn("Show Last Result", self.on_show_last))
        self.add_widget(self._btn("Settings", self.on_settings))
        self.add_widget(self._btn("Shutdown", self.on_shutdown))
        self.status = Label(text="", size_hint=(1, 0.2))
        self.add_widget(self.status)

    def _btn(self, text, cb):
        return Button(text=text, font_size="22sp", size_hint=(1, 0.15), on_press=cb)

    def on_start_scan(self, *_):
        self._flash("Starting scan...")
    def on_show_last(self, *_):
        self._flash("Opening last result...")
    def on_settings(self, *_):
        self._flash("Opening settings...")
    def on_shutdown(self, *_):
        self._flash("Shutting down...")
        Clock.schedule_once(lambda *_: os.system("sudo shutdown -h now"), 1.5)
    def _flash(self, msg):
        self.status.text = msg
        Clock.schedule_once(lambda *_: self._clear(), 2)
    def _clear(self):
        self.status.text = ""

class MenuScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(name="menu", **kwargs)
        self.container = Menu()
        self.container.opacity = 0  # start invisible; we’ll fade in
        self.add_widget(self.container)
        self.bind(on_enter=self._fade_in)

    def _fade_in(self, *_):
        Animation(opacity=1, d=0.25, t="out_quad").start(self.container)

# ---------------- App ----------------
class FFMenuApp(App):
    def build(self):
        self.title = "Forest-Finder"
        Window.multitouch_on_demand = True

        def _hide_cursor(*_):
            try: Window.show_cursor = False
            except Exception: pass
        Window.bind(on_draw=_hide_cursor)

        # CHANGED: fade between screens
        sm = ScreenManager(transition=FadeTransition(duration=0.28))
        sm.add_widget(SplashScreen())
        sm.add_widget(MenuScreen())
        sm.current = "splash"
        return sm

if __name__ == "__main__":
    FFMenuApp().run()
