"""Wall power per cell from the server's iDRAC, in the format of scripts/idrac_power.py.

Sum of the power supplies' input watts over Redfish (see that script for why the
chassis-level reading is not used on the XE7740). Credentials come from an env
file outside the repo. A failed sample is recorded and sampling continues; power
evidence never stops a measurement.
"""

import asyncio
import base64
import json
import ssl
import time
import urllib.request

POWER = "/redfish/v1/Chassis/System.Embedded.1/Power"


def load_env(path):
    env = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip('"').strip("'")
    missing = [k for k in ("IDRAC_HOST", "IDRAC_USER", "IDRAC_PASS") if k not in env]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    return env


def read_watts(env, timeout=10):
    token = base64.b64encode(f"{env['IDRAC_USER']}:{env['IDRAC_PASS']}".encode()).decode()
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE  # lab iDRAC, self-signed
    request = urllib.request.Request(
        f"https://{env['IDRAC_HOST']}{POWER}",
        headers={"Authorization": f"Basic {token}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        data = json.load(response)
    inputs = [
        float(p["PowerInputWatts"])
        for p in data.get("PowerSupplies", [])
        if p.get("PowerInputWatts") is not None
    ]
    if not inputs:
        raise RuntimeError("No PowerSupplies PowerInputWatts in Redfish Power resource")
    return sum(inputs)


async def sample_power(env_file, output, interval_s=1.0, read=read_watts):
    env = load_env(env_file)
    with open(output, "a") as out:
        while True:
            began = time.time()
            try:
                out.write(f"{began:.3f},{await asyncio.to_thread(read, env):.1f},\n")
            except Exception as exc:
                out.write(f"{began:.3f},,{type(exc).__name__}\n")
            out.flush()
            await asyncio.sleep(max(0.0, interval_s - (time.time() - began)))
