#!/bin/bash
# InferenceX-style sweeps on the XE7740, sequential, resumable.
# Runs from /data/capsim/inferencex on the box: nohup ./inferencex_queue.sh > queue.log 2>&1 &
set -u
cd "$(dirname "$0")"
run() { # workload config
  echo "$(date -Is) START $1 $2"
  python3 inferencex_xe7740.py sweep --workload "$1" --resume "$2"
  echo "$(date -Is) END $1 $2 rc=$?"
}
# 8k/1k is InferenceX's active scenario; the model with the most platforms first.
run 8k1k gpt-oss-120b-tp1
run 8k1k gpt-oss-120b-tp2
run 8k1k llama-70b-fp8-tp4
run 8k1k llama-70b-fp8-tp2
# 1k/1k: deprecated there since 2026-07 but the gpt-oss-120b rows on every platform are recent.
run 1k1k gpt-oss-120b-tp1
run 1k1k gpt-oss-120b-tp2
run 1k1k llama-70b-fp8-tp4
echo "$(date -Is) ALL DONE"
