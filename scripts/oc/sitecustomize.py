"""Registers the capsim OpenCompass add-ins in every Python process the
OpenCompass job starts (it puts this folder on PYTHONPATH and sets
CAPSIM_OC=1). Any other process is left alone."""
import os

if os.environ.get('CAPSIM_OC') == '1':
    import capsim_oc  # noqa: F401
