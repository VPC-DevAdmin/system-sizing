"""capsim — AI sizing and capacity engine."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("capsim")
except PackageNotFoundError:  # running from a checkout without install
    __version__ = "0.0.0+unknown"
