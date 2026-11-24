# Phase 2 server-view exports

Section **2.2 is implemented**. These exports use the existing 24-chunk corpus and already-seen questions for development. They do not establish a new topic-inference result or replace a fresh final test set.

## Run

```bash
python phase2_export.py results/phase2/new_export --histories 24
python phase2_views.py results/phase2/new_export/views/combined/attacker_view.json
```

For five-question histories, add `--history-length 5`. The default factor is 1.5, metric is cosine, and known-label budget is one record per topic. Choose an empty output directory; existing randomized exports are preserved.

The index is encrypted once per export. Every request, including repeated text, makes a fresh SDK query-encryption call and gets a fresh opaque event ID. Challenge bits use `secrets.randbelow(2)` independently of `--seed`, which controls workload draws only. Finite exports need not have equal observed counts of each challenge side.

## Files and boundaries

Each `views/<name>/attacker_view.json` has schema version 1, a common public setup, disclosed known labels, unlabeled context, and challenge histories. The [security specification](05_PHASE2_SECURITY_MODEL.md) defines the common setup and observer assumptions.

| View | Fields beyond event ID and result count |
|------|-----------------------------------------|
| `volume` | None |
| `ids` | Canonically sorted returned IDs |
| `ranked_ids` | Ranked returned IDs |
| `scores` | Ranked IDs and aligned top-k scores |
| `ciphertexts` | Encrypted query; encrypted index at file level |
| `combined` | Ciphertexts, ranked IDs, top-k scores, and full score vectors |

Record inventories are sorted by opaque ID. Full score vectors use that order. Scores rank highest first: cosine, raw dot product, or negative squared L2. Exact ties use ascending record ID. Zero-norm cosine inputs have score zero. SDK ciphertext components are exported as numeric arrays; no coordinate names, fitted vocabulary, or encryption metadata headers are included.

Give an attacker **one selected view file only**. Keep sibling views, the manifest, and `evaluator_only.json` separate. The evaluator file contains challenge bits, undisclosed record/query labels, source-question references, relevance labels, and plaintext-reference rankings. Keys, plaintext embeddings, and fitted embedding state are never exported.

## Replay and checks

`phase2_views.py` rejects extra fields at each schema boundary, invalid dimensions, nonfinite values, duplicate event IDs, and inconsistent counts. Ciphertext views recompute scores and rankings using only NumPy and `ciphertext_math.py`. Combined-view replay also checks saved scores and ranks against those recomputed values. Log-only replay validates its schema; it cannot independently verify search arithmetic without ciphertexts.

Tests cover actual SDK-output provenance, repeated-request calls, private-field rejection, tampered scores, deterministic tie handling, and replay from a folder without the SDK, corpus, key, or evaluator file. Phase 1 commands and artifacts retain their original schemas and ranking rules.

Saved development examples: `results/phase2/development_export/` contains 24 single-request histories; `results/phase2/development_sessions/` contains 12 five-request histories. Their replay reports pass. The [2.3 attack guide](07_PHASE2_ATTACKS.md) describes inference comparisons on these exports.
