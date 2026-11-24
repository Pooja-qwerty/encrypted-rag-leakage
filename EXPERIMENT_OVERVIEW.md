# Encrypted RAG leakage: experiment and results

## The question

When a client searches encrypted documents, can the search server infer whether the query is about **computer science (CS)** or **biology (Bio)**? We study this application-level topic question, not a formal break of the encryption scheme.

## What we built

The current corpus has **24 synthetic private chunks** (12 per topic). A client embeds questions and searches with plaintext retrieval or Cloaked encrypted retrieval. The server sees different permitted views: result counts, opaque returned IDs, ranks, scores, and encrypted vectors. It never receives the secret key or private text in the attacker input. A few record-topic labels may be disclosed before the challenge; all other truth stays in a separate evaluator file.

We ask 48 development questions and measure both **topic inference** and **search usefulness**. We also study five-query histories, where four questions share the majority topic. Phase 1 tests log attacks and fetch defenses; Phase 2 tests score and encrypted-vector attacks against the same search pipeline. The [threat model](docs/05_PHASE2_SECURITY_MODEL.md) gives the exact server view.

## Results so far

| Finding | What the current data says |
|---|---|
| Returned IDs leak topic clues | With 8 known labels out of 24 chunks, direct voting on unordered IDs averages about **75% question** and **80% session** accuracy for Cloaked factor 1.5. The balanced-topic guessing baseline is 50%. |
| Search quality changes with the factor | Factor 1.5 is close to plaintext relevance in this small study. Factor 7.2 lowers both relevance and the tested attack's success; it is not a privacy guarantee. |
| Fetch padding depends on the observer | Three dummy fetches lower the tested **fetch-log** question attack from about 75% to 65% at 2× record fetches. The **search server** still sees the real search results, so its tested attack stays about 75%. |
| Full download has a clear cost | A model that keeps the query local gives the tested fetch observer no topic signal, but fetches all 24 records: 8× the usual record count. This is a protocol model, not a deployed system. |
| Extra server clues are unresolved | On two small Phase 2 development exports, score and encrypted-vector attacks did **not consistently improve** on simple log attacks. This does not show those clues are harmless. |
| Exact ID matching needs more data | No opposite-topic histories shared an exact returned-ID signature: **0/24** single-query and **0/12** five-query histories. We cannot yet isolate score/vector leakage while IDs are fixed. |

## What these results mean

Encrypted retrieval can still expose useful clues through search behavior. The measured leakage depends on what the observer sees and how many labels it already knows. The current results are **small, synthetic development findings**: two topics, TF-IDF, author-defined relevance judgments, and reused source questions. They are not a claim about real-user accuracy or the encryption scheme's security definition.

## Next work

Use a larger public corpus, several topics, a pinned semantic embedding model, and fresh test questions. Then repeat the attacks and exact-ID matching across keys and settings. [Implementation plan](IMPLEMENTATION_PLAN.md) tracks that work.

## Reproduce / inspect

- Phase 1: `python evaluate_project.py` and `python defense_experiment.py`; see [study results](docs/02_RESULTS_SNAPSHOT.md) and [defense results](docs/04_DEFENSE_RESULTS.md).
- Phase 2: see the [export guide](docs/06_PHASE2_EXPORTS.md), [attack results](docs/07_PHASE2_ATTACKS.md), and [matching results](docs/08_PHASE2_MATCHED_IDS.md) for complete commands and saved outputs.
