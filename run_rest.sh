#!/bin/bash
# Remaining large-batch runs, launched concurrently (vLLM batches concurrent requests).
cd "$(dirname "$0")"
source .venv/bin/activate
reflexion run --n 500 --seed 42 --trials 3 --no-think --critic-feedback oracle --conditions actor_critic > run_oracle.log 2>&1 &
reflexion run --n 200 --seed 42 --trials 3 --no-think --strategy react --conditions actor_critic > run_react.log 2>&1 &
reflexion run --n 200 --seed 42 --trials 3 --no-think --strategy multihop --conditions actor_critic > run_multihop.log 2>&1 &
wait
reflexion report
