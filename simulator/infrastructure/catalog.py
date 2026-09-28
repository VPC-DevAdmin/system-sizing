"""Project definitions are separate from executable adapter readiness."""

import json
from importlib.resources import files


def projects():
    return json.loads(files(__package__).joinpath("catalog.json").read_text())["projects"]


def project(project_id):
    return next(p for p in projects() if p["id"] == project_id)


# Only adapters exercised by this implementation are allowed to launch.
EXECUTABLE = {"P01", "P02", "P03", "P04", "P05", "P06", "P10"}


def listing():
    return [
        dict(
            p,
            implementation_status="pilot_available"
            if p["id"] in EXECUTABLE
            else "implementation_pending",
        )
        for p in projects()
    ]
