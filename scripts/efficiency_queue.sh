#!/bin/bash
# Efficiency sweep queue for the XE7740 (docs/efficiency-sweep-plan.md).
# Runs from /data/capsim/effsweep on the box:  setsid nohup ./efficiency_queue.sh > effsweep.log 2>&1 < /dev/null &
# Waits for the GPUs (P11 no-cache rerun, InferenceX Qwen3.5 sweeps) and the Llama 2 download,
# then pass 1 = 1k/1k for every model, pass 2 = 8k/1k for every model that has the context.
set -u
cd "$(dirname "$0")"
busy() {
  pgrep -f queue7-nocache.sh >/dev/null || pgrep -f "p11_us_survey.py sweep" >/dev/null \
    || pgrep -f queue3.sh >/dev/null || pgrep -f "inferencex_xe7740.py sweep" >/dev/null
}
echo "$(date -Is) waiting for the GPUs and the Llama 2 download"
while busy; do sleep 120; done
while ! grep -q "DOWNLOAD DONE" dl_llama2.log 2>/dev/null; do
  grep -q "DOWNLOAD FAILED" dl_llama2.log 2>/dev/null && echo "$(date -Is) Llama 2 download failed; it will be skipped" && break
  sleep 120
done
# The driver resolves models through the main HF cache; the overflow hub is linked in like the other giants.
HUB=/data2/capsim/huggingface/hub; OVF=/data3/capsim/hf-overflow/hub
for m in "$OVF"/models--NousResearch--Llama-2-70b-chat-hf; do
  [ -d "$m" ] && [ ! -e "$HUB/$(basename "$m")" ] && ln -s "$m" "$HUB/$(basename "$m")" && echo "$(date -Is) linked $(basename "$m")"
done
echo "$(date -Is) GPUs free; starting"
pass() { # workload
  echo "$(date -Is) PASS $1 START"
  python3 efficiency_sweep.py sweep --workload "$1" --resume
  echo "$(date -Is) PASS $1 END rc=$?"
}
pass 1k1k
pass 8k1k
echo "$(date -Is) ALL DONE"
