# Privacy leakage in encrypted RAG (CS 666)

**Question:** Can retrieval logs reveal whether someone is asking CS or Biology questions, even when the vectors are encrypted?

We use synthetic teaching notes, Cloaked AI encrypted search, and an attacker that sees only allowed metadata. We measure query topics, session topics, and retrieval quality.

[Experiment and results](EXPERIMENT_OVERVIEW.md) · [Implementation plan](IMPLEMENTATION_PLAN.md) · [Threat model](docs/01_THREAT_MODEL.md) · [Slides and write-up](https://drive.google.com/drive/folders/1j-pPfdrZEiP8DRtYN23oMuouo9cy1THd)

## Current progress

| Piece | Status |
|-------|--------|
| Plaintext and Cloaked retrieval | Working, with matched data and queries |
| Log-only attacker | Working: direct voting and co-retrieval clustering |
| Known-label budgets | Compared 0, 1, 2, and 4 chunks per topic |
| Query and session evaluation | 48 questions; simulated five-question sessions. These questions are development data for Phase 2. |
| Retrieval relevance | Precision@3 and nDCG@3 with synthetic author-defined judgments |
| Defense and cost comparison | Working trace simulation with explicit observer controls |
| Phase 2 attacks and exact-ID matching | Development comparisons done; current exports have no exact cross-topic ID matches |
| Larger corpus, relevance review, and final report | WIP |

The current study finds stronger topic inference as the attacker knows more chunk labels. At factor 1.5 with four known chunks per topic, direct voting averages about **75% query accuracy** and **80% session accuracy**. See the results for the setup and limits.

## Setup

Use Python 3.10+ and run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a gitignored `.env` file:

```text
CLOAK_VECTOR_SECRET=<64 hex characters>
```

Generate a 32-byte key with `python -c "import secrets; print(secrets.token_hex(32))"`. Reusing it repeats the key configuration; SDK encryption still uses fresh randomness. Without a configured key, the demo generates a temporary one. Raw keys are never exported to results.

## Run the current evaluation

```bash
python evaluate_project.py results/project_evaluation
```

This uses 24 private chunks, 24 context query types, and 48 separate test questions. It compares both attackers on identical logs across three views: counts, unordered IDs, and ranked IDs. It varies known-label selection over ten draws and uses five keys at factors 1.5 and 7.2. Each simulated session contains four questions from one topic and one from the other.

The study saves `runs.csv`, `summary.json`, and each backend's logs. Relevance judgments are in `data/evaluation_queries.json`; they are synthetic, author-defined, and pending team review. Sessions reuse held-out traces and are not independent real-user sessions.

Give an attacker **only its view and the allowed session membership**. Keep `evaluator_only.json` separate: it contains undisclosed topic and relevance labels. Example attacker views and predictions are saved under each backend's `example_known_only/` and `example_clustering/` folders.

## Run the defense comparison

```bash
python defense_experiment.py results/project_evaluation results/defense_evaluation
```

This reuses saved traces and mixes real fetches with dummy record IDs. The client keeps its original top-three results. It can reduce inference by a separate fetch-log observer, while the search server still sees the real search results. A full-download model keeps the query local and fetches all 24 records. Costs count records, not measured bytes or latency. See the [defense results](docs/04_DEFENSE_RESULTS.md).

## Phase 2 development runs

Phase 2 now exports actual ciphertexts and scores into separate attacker views:

```bash
python phase2_export.py results/phase2/new_export
python phase2_views.py results/phase2/new_export/views/combined/attacker_view.json
python phase2_evaluate.py results/phase2/new_export results/phase2/new_attacks
```

These are development traces. Fixed log, score, and encrypted-geometry attacks run on the same exports, with private labels opened only after predictions are saved. The matched-ID check found no exact cross-topic matches in the current traces. Use new output directories to preserve earlier runs.

## Smaller demos

```bash
python cloak_spike.py
python leakage_experiment.py compare results/new_comparison --approximation-factor 1.5
python sweep_experiment.py results/new_sweep --keys 10
python -m unittest discover -s tests -v
```

The paired demo uses the earlier eight-question workload. The sweep compares factors 1, 1.5, 2, 4, and 7.2. Encrypted search defaults to cosine; `--metric dot_product` selects the earlier scoring method. Overlap measures agreement with plaintext; relevance metrics measure whether judged useful chunks were returned.

## Main files

| Path | Purpose |
|------|---------|
| `evaluate_project.py` | Label-budget, relevance, and session study |
| `defense_experiment.py` | Fetch padding, search-server controls, and a full-download model |
| `phase2_export.py`, `phase2_views.py` | Phase 2 ciphertext/score exports and isolated replay |
| `phase2_attack.py`, `phase2_evaluate.py` | Allowed-view attacks and paired development scoring |
| `phase2_match.py` | Exact-ID matching and conditional analysis |
| `data/evaluation_queries.json` | Separate test questions and relevance judgments |
| `log_attack.py` | Attacks saved logs; supports direct voting and clustering |
| `leakage_experiment.py` | Shared retrieval and leakage harness |
| `sweep_experiment.py` | Metric checks and factor/key sweep |
| `cloak/` | Cloaked SDK and retrieval helpers |
| `game3.py` | Earlier plaintext baseline with private-embedding calibration |
| `rag_leak_sim.py`, `game1*.py`, `game2*.py` | Earlier simulations and security games |

Game 3 remains a baseline with extra attacker knowledge. Phase 1 attacks receive logs and limited known labels. Phase 2 tests scores and ciphertext relationships; exact matched-ID feasibility is documented. Larger evaluation, judgment review, and the final write-up remain WIP.
