"""Quarto pre-render hook: copy pipeline outputs into the site."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from mode_probe.site_export import export  # noqa: E402

sys.exit(export())
