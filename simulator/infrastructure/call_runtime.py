"""Prepare an isolated speech image and launch only owned GPU resources."""

import json
import os
import subprocess
import threading
import uuid
from pathlib import Path

import httpx

from ..benchmarks.storage import sha256, write_json
from ..config import EngineConfig
from .calls import ASR_MODEL, ASR_REVISION, SPEECH_IMAGE
from .deployment import OwnedVllm


def prepare(path, corpus, base_image):
    build = path / "speech-build"
    build.mkdir()
    # Dependencies are isolated from the serving environment. Record the resolved
    # image ID and full dependency freeze; reuse that exact ID in every placement.
    (build / "Dockerfile").write_text(f"""FROM {base_image}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng python3-venv && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/speech && /opt/speech/bin/pip install faster-whisper==1.2.1 ctranslate2==4.8.2 nvidia-cublas-cu12 nvidia-cudnn-cu12
RUN ln -s /opt/speech/lib/python*/site-packages/nvidia /opt/speech/nvidia
RUN /opt/speech/bin/pip freeze > /opt/speech/packages.txt
ENTRYPOINT []
""")
    with (build / "build.log").open("w") as out:
        subprocess.run(
            ["docker", "build", "-t", SPEECH_IMAGE, str(build)],
            stdout=out,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=3600,
        )
    image = json.loads(
        subprocess.check_output(["docker", "image", "inspect", SPEECH_IMAGE], text=True)
    )[0]["Id"]
    model = Path("/data2/capsim/speech-models") / ASR_REVISION
    model.mkdir(parents=True, exist_ok=True)
    plan = dict(model=ASR_MODEL, revision=ASR_REVISION, corpus=corpus, port=19390)
    write_json(path / "speech-plan.json", plan)
    name = "capsim-infra-" + uuid.uuid4().hex
    journal = path / "block-prepare" / "replica-assets" / "resource.json"
    journal.parent.mkdir(parents=True, exist_ok=True)
    write_json(journal, {"owner": path.name, "name": name, "phase": "creating"})
    argv = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--label",
        "capsim.infrastructure=" + path.name,
        "--user",
        str(os.getuid()) + ":" + str(os.getgid()),
        "-e",
        "HF_HOME=/tmp/hf",
        "-v",
        str(model) + ":/model",
        "-v",
        str(path) + ":/evidence",
        "-v",
        str(Path(__file__).with_name("speech_runtime.py")) + ":/runtime.py:ro",
        image,
        "/opt/speech/bin/python",
        "/runtime.py",
        "prepare",
    ]
    try:
        with (path / "speech-prepare.log").open("w") as out:
            subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=3600)
    finally:
        owned = OwnedVllm(EngineConfig(), threading.Event(), path.name)
        owned._container_id = name
        owned.journal = journal
        owned.shutdown()
    assets = {
        "image": image,
        "model": ASR_MODEL,
        "revision": ASR_REVISION,
        "files": {str(p.relative_to(model)): sha256(p) for p in model.glob("*") if p.is_file()},
        "audio": {p.name: sha256(p) for p in (path / "audio").glob("*.wav")},
    }
    write_json(path / "speech-assets.json", assets)
    (path / "speech-packages.txt").write_bytes(
        subprocess.check_output(["docker", "run", "--rm", image, "cat", "/opt/speech/packages.txt"])
    )
    return image, model


class OwnedSpeech(OwnedVllm):
    def __init__(self, cfg, owner, image, model, path, device):
        super().__init__(cfg, threading.Event(), owner)
        self.image, self.model, self.path, self.device = image, model, path, device

    def health_check(self):
        try:
            return (
                httpx.get(f"http://127.0.0.1:{self.cfg.port}/health", timeout=2).status_code == 200
            )
        except httpx.HTTPError:
            return False

    def _build_docker_command(self):
        return [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            "placeholder",
            "--network",
            "host",
            "--runtime",
            "nvidia",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=" + str(self.device),
            "-e",
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            "-v",
            str(self.model) + ":/model:ro",
            "-v",
            str(self.path) + ":/evidence",
            "-v",
            str(Path(__file__).with_name("speech_runtime.py")) + ":/runtime.py:ro",
            self.image,
            "/bin/bash",
            "-c",
            "export LD_LIBRARY_PATH=/opt/speech/nvidia/cublas/lib:/opt/speech/nvidia/cudnn/lib:$LD_LIBRARY_PATH; "
            "exec /opt/speech/bin/python /runtime.py serve",
        ]
