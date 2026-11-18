#!/usr/bin/env python3
"""Read and stage BIOS settings on a Dell iDRAC over Redfish.

Credentials come from the same env-style file as idrac_power.py (mode 600,
never in the repo; the password is never printed).

    idrac_bios.py show                      # planner-relevant attributes, pending changes, open jobs
    idrac_bios.py set SubNumaCluster=Enabled [A=V ...]   # stage for the next reboot
    idrac_bios.py reboot                    # graceful restart so a staged change applies
    idrac_bios.py clear                     # delete the pending BIOS job

A staged change only takes effect on reboot. Values are checked against the
attribute registry before anything is sent.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from idrac_power import Idrac, load_env

SYSTEM = "/redfish/v1/Systems/System.Embedded.1"
BIOS = f"{SYSTEM}/Bios"
JOBS = "/redfish/v1/Managers/iDRAC.Embedded.1/Oem/Dell/Jobs"
ATTRS = ["SysProfile", "MemFrequency", "SubNumaCluster", "LogicalProc", "ProcTurboMode",
         "ProcCStates", "ProcC1E", "UncoreFrequency"]


def call(idrac, method, path, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    headers = {**idrac.headers, "Content-Type": "application/json"}
    req = urllib.request.Request(idrac.base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=idrac.ctx) as r:
            raw = r.read()
            return r.status, dict(r.headers), (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            msg = json.loads(raw)
        except ValueError:
            msg = raw.decode(errors="replace")
        sys.exit(f"{method} {path}: HTTP {e.code}\n{json.dumps(msg, indent=1)[:2000]}")


def open_jobs(idrac):
    out = []
    for m in idrac.get(JOBS, timeout=30).get("Members", []):
        j = idrac.get(m["@odata.id"], timeout=30)
        if j.get("JobState") not in ("Completed", "CompletedWithErrors", "Failed", "RebootCompleted"):
            out.append({k: j.get(k) for k in ("Id", "Name", "JobState", "PercentComplete", "Message")})
    return out


def show(idrac):
    cur = idrac.get(BIOS)["Attributes"]
    pending = idrac.get(f"{BIOS}/Settings").get("Attributes", {})
    power = idrac.get(SYSTEM).get("PowerState")
    print(json.dumps({"power": power,
                      "current": {a: cur.get(a) for a in ATTRS if a in cur},
                      "pending": pending,
                      "open_jobs": open_jobs(idrac)}, indent=1))


def stage(idrac, pairs):
    cur = idrac.get(BIOS)["Attributes"]
    reg = idrac.get(f"{BIOS}/BiosRegistry", timeout=60)
    entries = {e["AttributeName"]: e for e in reg.get("RegistryEntries", {}).get("Attributes", [])}
    want = {}
    for p in pairs:
        k, v = p.split("=", 1)
        if k not in cur:
            sys.exit(f"{k}: not a BIOS attribute on this system")
        allowed = [x.get("ValueName") for x in entries.get(k, {}).get("Value", [])]
        if allowed and v not in allowed:
            sys.exit(f"{k}={v}: allowed values are {allowed}")
        if cur[k] != v:
            want[k] = v
    if not want:
        print("already set; nothing staged")
        return
    status, headers, _ = call(idrac, "PATCH", f"{BIOS}/Settings",
                              {"Attributes": want, "@Redfish.SettingsApplyTime": {"ApplyTime": "OnReset"}})
    print(json.dumps({"staged": want, "status": status, "job": headers.get("Location")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=os.environ.get("IDRAC_ENV", "/mnt/xeonlab-data1/capsim/idrac/idrac.env"))
    ap.add_argument("cmd", choices=["show", "set", "reboot", "clear"])
    ap.add_argument("pairs", nargs="*")
    a = ap.parse_args()
    idrac = Idrac(load_env(a.env))
    if a.cmd == "show":
        show(idrac)
    elif a.cmd == "set":
        stage(idrac, a.pairs)
    elif a.cmd == "reboot":
        status, _, _ = call(idrac, "POST", f"{SYSTEM}/Actions/ComputerSystem.Reset", {"ResetType": "GracefulRestart"})
        print(json.dumps({"reset": "GracefulRestart", "status": status}))
    elif a.cmd == "clear":
        for j in open_jobs(idrac):
            if j["Id"].startswith("JID_"):
                call(idrac, "DELETE", f"{JOBS}/{j['Id']}")
                print(f"deleted {j['Id']}")
        show(idrac)


if __name__ == "__main__":
    main()
