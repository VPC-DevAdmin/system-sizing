"""Container-local speech endpoint; only frozen audio IDs are accepted."""

import importlib.metadata
import json
import subprocess
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    root = Path("/evidence")
    plan = json.loads((root / "speech-plan.json").read_text())
    if sys.argv[-1] == "prepare":
        from huggingface_hub import snapshot_download

        snapshot_download(
            plan["model"],
            revision=plan["revision"],
            local_dir="/model",
            allow_patterns=["*.json", "*.bin", "*.txt"],
        )
        for row in plan["corpus"]:
            out = root / "audio" / f"{row['id']}.wav"
            out.parent.mkdir(exist_ok=True)
            subprocess.run(
                [
                    "espeak-ng",
                    "-v",
                    row["voice"],
                    "-s",
                    str(row["speed"]),
                    "-w",
                    str(out),
                    row["reference"],
                ],
                check=True,
            )
        return
    from faster_whisper import WhisperModel

    model = WhisperModel(
        "/model", device="cuda", compute_type="float16", num_workers=2, cpu_threads=4
    )
    audio = {r["id"]: root / "audio" / f"{r['id']}.wav" for r in plan["corpus"]}
    lock = threading.Semaphore(2)
    (root / "speech-effective.json").write_text(
        json.dumps(
            {
                "model": plan["model"],
                "revision": plan["revision"],
                "compute_type": "float16",
                "beam_size": 5,
                "language": "en",
                "vad_filter": False,
                "num_workers": 2,
                "packages": {
                    n: importlib.metadata.version(n) for n in ["faster-whisper", "ctranslate2"]
                },
            },
            indent=2,
        )
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/transcribe" or not 0 < length < 1024:
                    raise ValueError("Invalid request")
                identity = json.loads(self.rfile.read(length))["id"]
                file = audio[identity]
                started = time.monotonic()
                with lock:
                    queued = time.monotonic() - started
                    segments, _ = model.transcribe(
                        str(file),
                        beam_size=5,
                        language="en",
                        vad_filter=False,
                        temperature=0,
                        condition_on_previous_text=False,
                    )
                    text = " ".join(s.text.strip() for s in segments)
                with wave.open(str(file)) as wav:
                    duration = wav.getnframes() / wav.getframerate()
                result = {
                    "text": text,
                    "audio_seconds": duration,
                    "server_s": time.monotonic() - started,
                    "asr_queue_s": queued,
                }
                payload = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            except Exception as exc:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode())

    ThreadingHTTPServer(("127.0.0.1", plan["port"]), Handler).serve_forever()


if __name__ == "__main__":
    main()
