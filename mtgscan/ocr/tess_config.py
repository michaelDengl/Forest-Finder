from __future__ import annotations
import os
from shutil import which


# Centralized Tesseract configuration helpers.
# Prefer fast models if present; otherwise fall back to default.


def default_config(psm: int = 7, extra: dict[str, str] | None = None) -> str:
cfg = ["--oem", "1", "--psm", str(psm), "-c", "user_defined_dpi=150"]
if extra:
for k, v in extra.items():
cfg += ["-c", f"{k}={v}"]
return " ".join(cfg)




def try_set_fast_models() -> None:
"""If tessdata_fast is installed, point TESSDATA_PREFIX to it.


Safe to call multiple times; it's a best-effort tweak.
"""
# Heuristic common paths on Debian/Raspberry Pi OS
candidates = [
"/usr/share/tesseract-ocr/5/tessdata_fast",
"/usr/share/tesseract-ocr/tessdata_fast",
"/usr/share/tesseract-ocr/4.00/tessdata_fast",
]
for p in candidates:
if os.path.isdir(p):
os.environ.setdefault("TESSDATA_PREFIX", p)
break


# Ensure tesseract exists (for nicer error messages elsewhere)
if which("tesseract") is None:
# Don't raise here; let pytesseract surface the error when used.
pass




# Auto-run at import so CLIs benefit without extra code.
try_set_fast_models()