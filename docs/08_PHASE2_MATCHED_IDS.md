# Phase 2.4: matched-ID analysis

Run the conditional analysis on an existing Phase 2 attack export:

```bash
python phase2_match.py results/phase2/development_export results/phase2/development_match_analysis
python phase2_match.py results/phase2/development_sessions results/phase2/development_sessions_match_analysis
```

The script checks unordered and ranked ID signatures separately. Each signature covers every request in a history. It freezes attack predictions before selecting opposite-topic pairs, then reports coverage and unmatched results. If pairs exist, it makes a blinded matched export and reruns the attacks on it.

## Current development result

| Export | Exact unordered ID matches | Exact ranked ID matches |
|---|---:|---:|
| Single-query (24 histories) | 0 / 24 | 0 / 24 |
| Five-query (12 histories) | 0 / 12 | 0 / 12 |

There are no matched histories to score, so this dataset cannot measure whether scores or encrypted geometry add inference when returned IDs are held constant. The saved analyses report zero coverage and retain the unconditional results: [single-query](../results/phase2/development_match_analysis/summary.json) and [five-query](../results/phase2/development_sessions_match_analysis/summary.json).

For any future exact unordered-ID group, an IDs-only attacker gets identical observations on both challenge sides when setup, context, and auxiliary knowledge are shared. Its conditional distinguishing advantage is therefore zero. This does not imply zero leakage from ranked IDs, scores, ciphertexts, or unmodeled channels. Ordered-ID matching is reported separately and is stricter.

This is a feasibility result on small, already-seen development traces, not final-test evidence. Do not broaden the matching rule and describe it as exact. Coarser matches need a separately named experiment and narrower interpretation.
