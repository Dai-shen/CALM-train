#!/bin/bash

while true; do
    python src/apply_delta.py --base 'decapoda-research/llama-7b-hf' --target './weights/vicuna-7b' --delta 'lmsys/vicuna-7b-delta-v1.1' && break
    sleep 1
done
