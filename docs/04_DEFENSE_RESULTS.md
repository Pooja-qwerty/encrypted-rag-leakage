# Defense results

## Setup

This trace simulation reuses the saved study: 48 questions, five keys at each Cloaked factor, and both existing attacks. The attacker knows four chunk labels per topic (eight of 24). We vary label selection over ten draws and dummy selection over three draws.

```bash
python defense_experiment.py results/project_evaluation results/defense_evaluation
```

## Results

Means for Cloaked factor 1.5 and direct known-label voting:

| Observable view | Query accuracy | Session accuracy | Records fetched |
|-----------------|----------------|------------------|-----------------|
| Fetch log, no padding | 75.2% | 80.2% | 1x |
| Fetch log, +3 dummies | 65.0% | 70.5% | 2x |
| Fetch log, +9 dummies | 57.6% | 62.7% | 4x |
| Fetch log, +21 dummies | 50.0% | 50.0% | 8x |
| Search server, +21 dummies | 75.2% | 80.2% | 8x |
| Full download, query stays local | 50.0% | 50.0% | 8x |

Without padding, each search fetches three records. Adding three dummies fetches six; adding nine fetches twelve. The client discards dummies and keeps the original ranking. Delivered-result relevance stays unchanged: mean nDCG@3 is 0.875.

## What the observer can see matters

The **fetch-only observer** sees a mixed, unordered batch. It cannot see the search response, ranking, query vector, or which records the client keeps. Dummies have no identifying flag. Padding weakens the tested attackers in this view.

The **search server** still computes and sees the real results. Its original ID-based attack stays unchanged, even if all records are fetched afterward. Padding therefore provides no demonstrated protection against that server.

The **full-download model** changes the protocol: send no query, fetch all 24 records, and search locally. Saved plaintext rankings model client selection. Its visible record sets are identical across questions, and the tested attacks score 50%. Local-result nDCG@3 is 0.878. This is a costly reference baseline, not a deployed protocol.

## Costs and limits

Costs count records per query, relative to three unpadded fetches. They do not measure actual bytes, latency, or client computation. Sessions reuse traces. Results cover the two existing attacks; a stronger padding-aware attacker or repeated-query analysis may do better against partial padding.

[Summary](../results/defense_evaluation/summary.json) · [Individual evaluations](../results/defense_evaluation/runs.csv) · [Settings](../results/defense_evaluation/settings.json)

Each scenario folder saves an observer view, separate evaluator data, and example predictions. Session membership is saved once per source configuration. Keep evaluator files private from the attacker.
