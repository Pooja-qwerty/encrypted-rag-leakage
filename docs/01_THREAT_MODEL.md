# Threat model

This document describes Phase 1. Phase 2's expanded observations and challenge experiment are specified [separately](05_PHASE2_SECURITY_MODEL.md).

## Parties and goal

The client holds the private text and encryption keys. The search server runs retrieval correctly but may be curious about query topics. We study a restricted observer of its logs: can it guess CS vs Bio?

The full server also sees encrypted vectors, encrypted queries, and search scores. Our controlled log views isolate metadata leakage; they do not model everything the server can learn.

## Attacker inputs

The main attacker receives only `attacker_view.json`:

- Unlabeled context logs from earlier searches.
- Challenge logs with opaque query IDs.
- A fixed set of known `(chunk ID, topic)` pairs, selected before clustering.
- The leakage level and the two possible topics: CS and Bio.
- For session experiments, a separate `session_view.json` links query IDs to opaque session IDs. This linkage is an explicit extra observable.

The single-run default budget is **one known chunk per topic**. The broader study varies 0, 1, 2, and 4 known chunks per topic over ten label-selection seeds. These pairs model prior knowledge about a few records; a public corpus alone does not supply this mapping. Selection is random within each topic, independent of the logs. Budgets are nested within each seed. Some known chunks may never appear in the context logs.

The attacker receives no text, embeddings, keys, generation seed, or other topic labels. It cannot submit queries. The harness saves undisclosed labels in `evaluator_only.json` for scoring afterward.

## Log views

| View | One search reveals |
|------|--------------------|
| Volume | Number of returned chunks |
| AP-set | Count and unordered opaque chunk IDs |
| AP-ordered | Count and ranked opaque chunk IDs |

Records are shuffled and assigned random opaque IDs before retrieval. Query IDs also carry no topic information. Repeated queries can still have identical result sets.

## Attack and scoring

`log_attack.py` compares direct voting from known labels with co-retrieval clustering. Clustering uses context logs only; known labels name its two groups by majority vote. Tied or unnamed groups remain unknown. Challenge IDs vote for a query topic, with rank weights in AP-ordered. A tie or lack of evidence falls back to CS and is recorded as having no signal.

Session inference pools chunk votes across linked queries; it does not fit on session labels. Synthetic sessions have five distinct queries with a 4:1 majority topic. Session labels are kept with the evaluator. Sessions reuse held-out retrieval traces and are not independent real-user sessions.

The evaluator scores every challenge, including fallbacks. It reports accuracy and signed advantage: `Pr(guess Bio | Bio) - Pr(guess Bio | CS)`. Chunk-label accuracy excludes disclosed known labels and reports how many other chunks were labeled.

## Current scope

The defense experiment separates three views. A **fetch-only observer** sees real and dummy IDs mixed in an unordered batch, with no search ranking, query vector, or filtering decisions. The **search-server control** retains the original real-result IDs; dummy fetches do not hide these. These views must not be treated as the same adversary.

A **full-download protocol model** sends no query to the server: the client fetches every record and selects results locally. Saved plaintext rankings model that local selection. This changes where search runs and incurs full-corpus transfer and local work; it is a trace model, not a deployed protocol. Timing and byte sizes are outside the experiment.

The broader experiment uses TF-IDF retrieval, 24 context query types, and 48 separate test questions. The corpus still has 24 chunks. Earlier paired demos retain eight test questions. Both backends share the data, query order, opaque IDs, and known-label budget. The Cloaked path encrypts the index once and prepares an encrypted query for each search; only returned IDs reach the attacker. Plaintext reference rankings, relevance judgments, and overlap scores stay in the evaluator file.

Encrypted search now defaults to cosine similarity; earlier saved paired results used dot products. The sweep varies approximation factor and derived key while keeping the workload and known labels fixed. Known-neighbor checks use tiny noise only as a correctness control.

Repeated context searches are repeated observations, not new query types. Relevance judgments are synthetic and author-defined, pending team review. Larger corpus evaluation, stronger padding-aware attacks, and final reporting are WIP. Small-demo results are not a security proof.

`game3.py` remains a separate baseline with extra knowledge: its calibration reads private plaintext embeddings. The SAP/RAP transforms in `rag_leak_sim.py` are separate simulations.
