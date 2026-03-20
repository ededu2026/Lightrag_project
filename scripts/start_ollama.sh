#!/bin/sh

set -eu

ollama serve &
sleep 5
ollama pull "${OLLAMA_MODEL:-qwen2.5:3b}"
wait
