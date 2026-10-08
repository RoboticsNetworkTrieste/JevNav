#!/usr/bin/env bash
set -euo pipefail

CACHE="${JEVNAV_CACHE:-$HOME/.cache/jevnav}"
LLAMA_SERVER="${CLM_LLAMA_SERVER:-$CACHE/clm/runtime/llama-b11081/llama-server}"
ENCODER="${CLM_ENCODER:-$CACHE/clm/models/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf}"
HEAD="${CLM_HEAD:-$CACHE/clm/heads/CLM_v0.1-8B.pt}"
CLM_SERVE="${CLM_SERVE:-$CACHE/venvs/clm/bin/clm-serve}"
ENCODER_PORT="${CLM_ENCODER_PORT:-8090}"
CLM_PORT="${CLM_PORT:-8700}"
CONTEXT="${CLM_CONTEXT:-4096}"
LOG="$CACHE/clm/llama-server.log"

for file in "$LLAMA_SERVER" "$ENCODER" "$HEAD" "$CLM_SERVE"; do
    [[ -e "$file" ]] || { echo "serve-clm: missing $file (see README, Model server)" >&2; exit 2; }
done

"$LLAMA_SERVER" -m "$ENCODER" --embeddings --pooling last \
    -c "$CONTEXT" -b "$CONTEXT" -ub "$CONTEXT" -np 1 -ngl 99 \
    --host 127.0.0.1 --port "$ENCODER_PORT" >"$LOG" 2>&1 &
encoder=$!
trap 'kill "$encoder" 2>/dev/null || true' EXIT INT TERM

echo "serve-clm: loading the Qwen3-8B encoder on :$ENCODER_PORT (log: $LOG)"
until curl -sf "http://127.0.0.1:$ENCODER_PORT/health" >/dev/null; do
    kill -0 "$encoder" 2>/dev/null || { tail -n 20 "$LOG" >&2; exit 1; }
    sleep 0.5
done

"$CLM_SERVE" --emb-url "http://127.0.0.1:$ENCODER_PORT/v1/embeddings" --ckpt "$HEAD" \
    --device cpu --host 127.0.0.1 --port "$CLM_PORT" "$@"
