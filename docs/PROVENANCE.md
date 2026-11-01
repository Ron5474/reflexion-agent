# Provenance of the December 2025 results

This repo is a port of `notebooks/legacy_gemini_run_2025-12.ipynb`, the notebook behind the CMPE 252
final report (Hriday Ampavatina and Ronak Patel, San Jose State University, Dec 2025). The port keeps
the prompts, the 50-question seed-42 sample, the chunking parameters, the relaxed exact-match rule and
the up-to-3-trial loop, so a re-run here is comparable to the notebook.

## What the notebook reported

| Condition | Correct / 50 | Accuracy |
|---|---|---|
| Full context (no RAG, no memory) | 27 | 54% |
| RAG only | 31 | 62% |
| Actor-Critic (RAG + reflection memory) | 38 | 76% |
| Actor strategy sweep: CoT / ReAct / Multi-hop | 37 / 39 / 40 | 74% / 78% / 80% |

## Three things the audit found, and what this repo changes

1. **Model label.** Every LLM call in the notebook goes to `gemini-2.5-flash`. The summary table
   labels the rows "Qwen-2.5-7B". The numbers above are Gemini numbers. This repo records the model
   name that actually answered in `config.json` and `summary.json` of every run.
2. **Baseline contamination.** 22 of the 100 baseline calls failed with HTTP 429 quota errors. The
   helper returned the string "Error: Could not get a response", which scored as a wrong answer.
   So the 54% and 62% baselines are lower bounds and the "+14 over RAG" gain is not trustworthy.
   This repo retries with backoff, raises on failure, records errors separately, excludes them from
   accuracy, and fails the run unless `--allow-errors` is passed.
3. **Label leakage in the critic.** The critic prompt included `Correct answer: {gold}`. Reflections
   written with the gold answer were stored in memory and shown to the actor on retry for the same
   question. The Reflexion paper's critic only receives the binary success signal. This repo defaults
   to `--critic-feedback binary` and keeps `oracle` available so the notebook's setup can be reproduced
   and the two compared.

Until a clean run exists, the only defensible claims from the notebook are the Actor-Critic 76%
(no errors in that cell) and the strategy sweep 74 to 80% (no errors in that cell), both under the
relaxed scorer, both with oracle critic feedback, both with Gemini 2.5 Flash.
