# Phase 2 attacks and development results

## Run

```bash
python phase2_evaluate.py results/phase2/development_export results/phase2/new_attacks
```

For one isolated attacker:

```bash
python phase2_attack.py results/phase2/development_export/views/ciphertexts/attacker_view.json /tmp/predictions.json --method geometry
```

Use an empty output directory. The runner checks that all six views project from the same trace and known-label map. It saves every attack/control prediction **before opening evaluator labels**.

## Fixed attack rules

- **Direct/co-retrieval:** Phase 1 voting, with rank weights where rank is visible. Co-retrieval labels are fitted on context logs only.
- **Score weighting:** softmax weights from top-k scores. Scale is the standard deviation of unlabeled context top-k scores, or 1 if unavailable/constant. Use known labels or the same co-retrieval map.
- **Known-anchor geometry:** compare mean query similarity to the known records of each topic, using all available ciphertext-derived scores. No private plaintext vectors are used.
- **Neighbor graph:** build a symmetric five-neighbor graph of encrypted records. Propagate known labels for 50 steps with weight 0.85 and neutral priors, keeping disclosed labels clamped. Classify each query from five nearby records' topic weights.
- **Combined:** average the score/co-retrieval evidence and graph evidence with fixed 50/50 weights.

Rules were fixed before scoring these exports. Only context, encrypted index, and disclosed labels affect fitted state. Challenge histories do not train the model. Pool evidence within each history; ties fall back to topic 0. Ciphertext and combined views produce identical predictions for the same method because the logs/scores are derivable.

## Results

Both exports disclose **two record labels: one CS and one Bio**. The table reports **balanced accuracy**, averaging success separately over each challenge side; its constant-guess baseline is 50%.

| Method | Single-query histories | Five-query histories |
|--------|------------------------|----------------------|
| IDs: direct voting | 72.4% | 56.2% |
| IDs: co-retrieval | 72.4% | 50.0% |
| Ranked IDs: direct voting | 72.4% | 56.2% |
| Top-k scores: direct weighting | 72.4% | 56.2% |
| Top-k scores: co-retrieval weighting | 72.4% | 50.0% |
| Ciphertexts: known anchors | 70.3% | 50.0% |
| Ciphertexts: neighbor graph | 66.4% | 37.5% |
| Combined: fixed score/graph mixture | 71.0% | 56.2% |

The single-query export has 24 histories (11 CS-side, 13 Bio-side); the session export has 12 (4 CS-side, 8 Bio-side). Raw accuracy and realized-majority baselines are also saved. These exports have separately randomized index realizations and selected known records; differences between columns do not isolate history length.

**Observation:** Scores do not improve the single-query baseline here. Geometry and the combined method do not consistently outperform log-only attacks. This is a valid negative development result, not evidence that extra observations can never help.

## Controls and limits

Each export has 84 method/view/control evaluations: original labels, zero labels, and swapped topic names. Zero-label models use their declared fallback and score balanced 50%. Swapping labels tests orientation sensitivity; it may reverse the signed advantage and is not a null significance test. Negative predictions are not flipped using evaluator labels.

These are small, already-seen development workloads, one realized index per export, with repeated source questions. No confidence interval or final-test security claim is made. Fit timings include input validation/replay and are not a final performance benchmark. Matched-ID analysis (2.4) and larger evaluation (2.5) remain WIP.

- [Single-query summary and paired wins/losses](../results/phase2/development_attacks/summary.json)
- [Five-query summary and paired wins/losses](../results/phase2/development_session_attacks/summary.json)
- Each result folder contains `runs.csv` and predictions for every view, method, and control.

All 46 tests pass, including score-sensitive decisions, zero-label fallbacks, ciphertext/combined equivalence, challenge-independent fitting, and isolated inference without SDK or evaluator dependencies.
