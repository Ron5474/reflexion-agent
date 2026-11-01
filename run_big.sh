#!/bin/bash
# Larger batches. Seed-42 sampling is shuffle-then-prefix, so the first 50 of n=500 are the notebook's 50.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
reflexion run --n 500 --seed 42 --trials 3 --no-think --workers 8
reflexion run --n 500 --seed 42 --trials 3 --no-think --critic-feedback oracle --conditions actor_critic
for s in react multihop; do
  reflexion run --n 200 --seed 42 --trials 3 --no-think --strategy "$s" --conditions actor_critic
done
reflexion report
