# Implementation plan

**Goal:** Measure what a search server can learn about query topics during encrypted RAG retrieval. [Experiment and results](EXPERIMENT_OVERVIEW.md) gives the short story; `docs/` holds details. `[x]` means done; `[ ]` means WIP.

## Phase 1 — Small controlled study

- [x] Build 24 synthetic CS/Bio chunks, plaintext retrieval, and Cloaked encrypted retrieval with opaque record IDs.
- [x] Define a restricted log attacker with limited known record labels; keep keys, text, and private labels in the evaluator.
- [x] Compare count, unordered ID, and ranked ID views; test direct voting, co-retrieval clustering, and five-query sessions.
- [x] Measure attack success and retrieval quality across known-label budgets, keys, and approximation factors.
- [x] Model dummy fetches and full download, with separate fetch-observer and search-server results.
- [x] Save reproducible results and document the threat model. Keep the earlier plaintext Game 3 as an extra-knowledge baseline.
- [ ] Finish the Phase 1 write-up and have the team review relevance judgments.

## Phase 2 — Search-server observations

**Question:** Do ranks, scores, or encrypted vectors help infer a topic beyond returned IDs? The same Cloaked search pipeline remains in use.

### 2.1 Define the experiment — done

- [x] Specify a passive server, two topic choices, fixed auxiliary knowledge, allowed views, and signed/absolute advantage. Treat the existing questions as development data. [Threat model](docs/05_PHASE2_SECURITY_MODEL.md).

### 2.2 Export observations — done

- [x] Export six versioned views: count, IDs, ranked IDs, scores, ciphertexts, and combined. Keep evaluator truth separate and replay scores/ranks from real SDK ciphertexts. [Export guide](docs/06_PHASE2_EXPORTS.md).

### 2.3 Run stronger attacks — done on development data

- [x] Compare log baselines with score weighting, encrypted-anchor comparison, neighbor-graph propagation, and a fixed combination. Save predictions before opening truth labels; include zero-label and swapped-label controls. [Attack results](docs/07_PHASE2_ATTACKS.md).

### 2.4 Match returned IDs — done as a feasibility check

- [x] Match opposite-topic histories by exact per-request ID sets; check ranked IDs separately. Balance sides, hide pair labels from the attacker, and report coverage and unmatched results. Current exports have **zero exact cross-topic matches**, so no conditional attack comparison is possible yet. [Matched-ID analysis](docs/08_PHASE2_MATCHED_IDS.md).

### 2.5 Larger, fresh evaluation — WIP

- [ ] Build roughly 1,000 public-document chunks across several topics; scale toward 5,000 if runtime permits.
- [ ] Pin and cache a semantic embedding model; keep TF-IDF as a baseline.
- [ ] Separate context, development, and fresh test questions; review relevance labels.
- [ ] Test 0%, 1%, and 5% known-label settings, noisy labels, topic mixtures, and different session lengths.
- [ ] Vary keys and approximation factors. Report retrieval quality, attack accuracy/advantage, exact-match coverage, uncertainty, runtime, and memory.
- [ ] Keep attack training inside each index/key setting; treat cross-key transfer as a separate experiment.

### 2.6 Final analysis — WIP

- [ ] Explain what IDs, rank, scores, vectors, and history each add, using paired comparisons where data supports them.
- [ ] State what identical observations prove, what attack results show, and what remains untested. Keep the fetch-observer defense separate from the search-server analysis.
- [ ] Publish concise results, schemas, settings, and reproduction commands.

**Optional:** Test search-pattern leakage and stronger padding-aware fetch attacks. A new encryption scheme, PIR, or live database is outside this plan.
