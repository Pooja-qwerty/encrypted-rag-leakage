# Results snapshot

## Current study

The study uses 24 synthetic chunks, 24 context query types, 48 separate test questions, and top-3 retrieval. It compares direct known-label voting and co-retrieval clustering on the same logs. Known labels vary over ten selections. Cloaked factors 1.5 and 7.2 each use five keys.

```bash
python evaluate_project.py results/project_evaluation
```

There are **11 retrieval configurations** and **2,640 attack evaluations** across label budgets, methods, and three log views. These reuse the same 48 test questions; they are not 2,640 independent workloads. Tables below use unordered IDs (AP-set).

## 1. Known labels matter

Mean query accuracy for direct voting; the balanced-topic baseline is 50%:

| Known chunks per topic | Plaintext | Cloaked 1.5 | Cloaked 7.2 |
|------------------------|-----------|-------------|-------------|
| 0 | 50.0% | 50.0% | 50.0% |
| 1 | 56.2% | 55.8% | 52.1% |
| 2 | 61.7% | 60.3% | 54.2% |
| 4 | 79.0% | 75.2% | 59.9% |

Four known chunks per topic means **eight known chunks out of 24**. Stronger prior knowledge makes this attacker more effective on both backends. Zero-label and count-only controls stay at 50%.

## 2. Clustering gives mixed benefits

At factor 1.5, clustering averages 58.9%, 64.8%, and 74.9% query accuracy for budgets 1, 2, and 4. Direct voting averages 55.8%, 60.3%, and 75.2%. Clustering helps at the smaller budgets here, but does not consistently beat the simpler attacker. On plaintext, it performs worse at every nonzero budget tested.

## 3. Relevance and session topics

Sessions contain five distinct test queries: four from the majority topic and one from the other. The observer is explicitly allowed to link those queries to a session. There are 20 simulated sessions per majority topic, resampled from the same saved traces.

| Backend | Precision@3 | nDCG@3 | Session accuracy, direct voting, budget 4 |
|---------|-------------|--------|-----------------------------------------|
| Plaintext | 34.7% | 0.878 | 82.5% |
| Cloaked 1.5 | 35.0% | 0.875 | 80.2% |
| Cloaked 7.2 | 22.4% | 0.525 | 66.2% |

Relevance at factor 1.5 stays close to plaintext under these judgments, while topic inference remains possible. Factor 7.2 lowers relevance and weakens this attacker, but its mean session accuracy still exceeds chance.

Only one or two chunks are labeled relevant per question, so maximum average precision@3 under these labels is **41.7%**. nDCG accounts for this and rewards relevant chunks appearing earlier.

## Limits and saved files

Relevance labels are synthetic and author-defined, pending team review. Test questions come from the same small corpus, with two authored questions per source chunk. Sessions share traces. Means, standard deviations, and ranges describe variation across known-label draws and keys; they are not confidence intervals over independent real users.

- [Summary and paired clustering gains](../results/project_evaluation/summary.json)
- [All attack evaluations](../results/project_evaluation/runs.csv)
- [Questions and relevance judgments](../data/evaluation_queries.json)
- Backend folders contain original logs, separate evaluator labels, session membership, and replayable examples.

The earlier [factor sweep](../results/search_sweep/summary.json) used eight test questions. Its [30 known-neighbor checks](../results/search_sweep/metric_validation.json) passed. Game 3 remains a separate plaintext baseline with private-embedding calibration.

**Interpretation:** The earlier near-chance result was specific to a limited attacker and workload. The expanded study shows query and session topic leakage when the attacker knows more chunk labels, including at a configuration that preserves relevance. The [defense experiment](04_DEFENSE_RESULTS.md) measures fetch padding, observer limits, and transfer cost. Larger corpus evaluation remains WIP.
