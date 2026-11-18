#!/usr/bin/env python3
"""Sample server input (wall) power from a Dell iDRAC over Redfish.

Credentials come from an env-style file (default /data/capsim/idrac/idrac.env,
mode 600, never in the repo):

    IDRAC_HOST=100.67.151.208
    IDRAC_USER=readonly
    IDRAC_PASS=...

    idrac_power.py --once                 # one reading, prints JSON
    idrac_power.py --interval 1 -o power.csv   # epoch,psu_input_sum_w,chassis_w until killed

Reads Chassis/System.Embedded.1/Power and sums PowerSupplies[].PowerInputWatts:
server input (wall) power, what MLPerf power results and InferenceX's power
model call the system or utility side of one server. See Idrac.watts() for why
the chassis-level readings are not used.
"""

import argparse
import base64
import json
import os
import ssl
import sys
import time
import urllib.request

CHASSIS = "/redfish/v1/Chassis/System.Embedded.1"


def load_env(path):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in ("IDRAC_HOST", "IDRAC_USER", "IDRAC_PASS") if k not in env]
    if missing:
        sys.exit(f"{path}: missing {', '.join(missing)}")
    return env


class Idrac:
    def __init__(self, env):
        self.base = f"https://{env['IDRAC_HOST']}"
        token = base64.b64encode(f"{env['IDRAC_USER']}:{env['IDRAC_PASS']}".encode()).decode()
        self.headers = {"Authorization": f"Basic {token}", "Accept": "application/json"}
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE  # lab iDRAC, self-signed
        self.source = None

    def get(self, path, timeout=10):
        req = urllib.request.Request(self.base + path, headers=self.headers)
        with urllib.request.urlopen(req, timeout=timeout, context=self.ctx) as r:
            return json.load(r)

    def watts(self):
        """Server input power in W: the sum of the PSUs' PowerInputWatts.

        On this XE7740 the chassis-level readings (EnvironmentMetrics PowerWatts
        and Power PowerControl PowerConsumedWatts, both from the
        SystemBoardPwrConsumption sensor) report only PS1's input most of the
        time (~745 W against ~3,200 W of PSU inputs) and the true total only
        every ~minute, so they are not usable. Per-PSU input watts add up to the
        occasional correct chassis total. The chassis reading is kept as a
        secondary field for the record."""
        d = self.get(CHASSIS + "/Power")
        psus = [p.get("PowerInputWatts") for p in d.get("PowerSupplies", [])]
        inputs = [float(w) for w in psus if w is not None]
        if not inputs:
            raise RuntimeError("no PowerSupplies PowerInputWatts in Redfish Power resource")
        self.source = f"PowerSupplies[{len(inputs)}].PowerInputWatts"
        self.last_chassis_w = d["PowerControl"][0].get("PowerConsumedWatts")
        self.last_psu_w = inputs
        return sum(inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=os.environ.get("IDRAC_ENV", "/data/capsim/idrac/idrac.env"))
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("-o", "--output", help="CSV path (default stdout)")
    a = ap.parse_args()
    idrac = Idrac(load_env(a.env))
    if a.once:
        t = time.time()
        w = idrac.watts()
        print(json.dumps({"epoch": round(t, 3), "watts": w, "source": idrac.source,
                          "psu_input_w": idrac.last_psu_w, "chassis_w": idrac.last_chassis_w,
                          "latency_s": round(time.time() - t, 3)}))
        return
    out = open(a.output, "a") if a.output else sys.stdout
    try:
        while True:
            t0 = time.time()
            try:
                w = idrac.watts()
                out.write(f"{t0:.3f},{w:.1f},{idrac.last_chassis_w}\n")
            except Exception as e:  # keep sampling through transient iDRAC hiccups
                out.write(f"{t0:.3f},,{type(e).__name__}\n")
            out.flush()
            time.sleep(max(0.0, a.interval - (time.time() - t0)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
