#!/bin/bash
# Full evaluation set. ~1,000 chat calls. Each run fails loudly on any LLM error.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
reflexion run --n 50 --seed 42 --trials 3 --no-think                                   # binary critic, 3 conditions
reflexion run --n 50 --seed 42 --trials 3 --no-think --critic-feedback oracle --conditions actor_critic   # notebook setup
for s in react multihop; do
  reflexion run --n 50 --seed 42 --trials 3 --no-think --strategy "$s" --conditions actor_critic
done
reflexion report
