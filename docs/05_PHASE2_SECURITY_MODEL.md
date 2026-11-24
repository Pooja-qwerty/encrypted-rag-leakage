# Phase 2 security experiment

**Status:** 2.1 specified; 2.2 exports and 2.3 development attacks implemented. Matched-ID and final-test evaluation are WIP. See the [export guide](06_PHASE2_EXPORTS.md) and [attack results](07_PHASE2_ATTACKS.md).

**Question:** Do ciphertext relationships and similarity scores help infer query topics beyond returned record IDs, given the same prior knowledge?

## Parties and assumptions

The client holds text, plaintext embeddings, and keys. The server stores encrypted vectors and searches correctly. The attacker passively observes a declared part of this server view.

Use a fixed database, one tenant, and one active key per replicate. Exclude updates, key rotation, malicious responses, and attacker-chosen query/encryption oracles. Other keys form separate replicates.

## Common knowledge and allowed views

Common public setup: topic pair, opaque record inventory, database/vector dimensions, embedding algorithm or public model identifier, metric, approximation factor, top-k, and history shape. Private TF-IDF vocabulary and weights stay hidden.

Every view also receives the same known `(chunk ID, topic)` pairs and the same side-independent context history, projected to its allowed fields.

| View | Per-request and index observations |
|------|-------------------------------------|
| `volume` | Result count |
| `ids` | Count and unordered returned IDs |
| `ranked_ids` | Count and returned IDs in rank order |
| `scores` | Ranked IDs and their top-k ciphertext similarity scores |
| `ciphertexts` | Encrypted index vectors mapped to IDs and encrypted query vectors |
| `combined` | Ciphertexts plus the derived logs and full index/query score vectors |

All views receive the same requests in the same order. Session membership, when disclosed, is common to every view. Compare each history length separately.

Scores and ranks follow from ciphertexts and the recorded search rule. Thus `combined` does not add independent information to `ciphertexts` under this model; differences between attacks on them concern computation or representation. The main additional-observation comparison is against the restricted log views.

Use fresh opaque event IDs, including for repeats. Supply no repeated-text token; attackers may infer repetition from allowed observations. Timing, message lengths, headers, and client identities are outside these views. `combined` models selected server observations, not every side channel.

## Auxiliary knowledge and development split

Known chunk labels are selected before the challenge side, independently of search results. Publish their number, fraction, selection rule, and any label-error rate. Reuse exactly the same selection across views. Perfect and noisy-label settings are separate experiments.

Fit from known labels and unlabeled context only. Withhold query text, query labels, relevance judgments, plaintext vectors, private fitted embedder, keys, and generation seeds. Public benchmark documents serve as private-role data, not automatic auxiliary input.

The existing 48 questions are **development material**, since we have examined their results. Freeze attacks before opening fresh final test questions. Keep paraphrase families in one split. Final queries can reference the same index, but cannot recycle development examples.

Supervised query-label training needs a separate setting and equal budgets across baselines. Record labels alone do not grant labeled queries. Fit within each key/index setting; cross-key transfer is separate.

## Challenge experiment

The challenger is the trusted experiment harness; its private state never goes to the attacker.

1. Fix the database, embedder, metric, factor, top-k, topic pair `(topic0, topic1)`, history length, session layout, and query-sampling rules. Separate context, development, and final query families.
2. Generate a key, shuffle records, assign opaque IDs, and encrypt the index once. Select auxiliary labels and generate an unlabeled context history independent of the future challenge side.
3. Fit/freeze the attacker using only its allowed setup, context, and auxiliary knowledge.
4. Sample a fair bit `b`. Draw a final history from the predeclared distribution for `topic_b`. Both worlds have equal request counts, session layouts, and fixed top-k result counts.
5. Embed/encrypt each request using the actual SDK with fresh randomness. Provide the chosen view of the challenge history. The attacker may use its observed prefix, but receives no feedback or extra queries.
6. The attacker outputs one bit for the history. The evaluator opens `b` afterward and scores the guess. Retain every trial, including ties and fallbacks.

Start with one-request histories. The session variant uses five distinct questions: four from `topic_b` and one from the other topic, in shuffled order. Both worlds use this rule. Larger multi-topic datasets use predeclared pairwise comparisons; report each pair and their macro-average. Multiclass accuracy is a separate metric.

Pair candidates using the same setup, but expose only the selected candidate. Freeze/clone fitted state across sides and views. Draws from one key/index are dependent; vary keys, label selections, and query families and report those sampling units.

## Scoring and controls

Use `0 = topic0` and `1 = topic1`:

```text
delta = Pr[guess = 1 | b = 1] - Pr[guess = 1 | b = 0]
Adv   = abs(delta)
```

Report both the signed gap `delta` (Phase 1 convention) and absolute distinguishing advantage `Adv`, plus accuracy, fallback rate, and uncertainty. Under balanced sides, accuracy is `(1 + delta) / 2`. Never flip predictions using final-test labels. A fixed fallback to `topic0` is declared in advance and remains in the denominator.

Compare every view on the same realized SDK traces and auxiliary selection. Keep constant-count, majority, zero-known-label, and label-permutation controls. Compare score/ciphertext attacks against the Phase 1 attacks with the same topic pair, context, and history length. Report relevance and runtime alongside inference results. Finite-sample nonzero advantage alone is not a security conclusion.

## Matched-ID diagnostic and claim boundary

Section 2.4 will separately select cross-topic held-out traces with identical ID signatures, before inspecting attack success. Context, auxiliary input, public metadata, and challenge-side balance must match. Pair identities and topic labels remain evaluator-only. Report matching coverage and unconditional results.

For an exactly matched unordered-ID view with identical side-independent setup/context and opaque event IDs independent of the side, the two observation distributions are identical. Its theoretical distinguishing advantage is zero. This statement does not apply to ranked IDs, scores, or ciphertexts unless those observations also match. Conditioned trace selection is a diagnostic, not the unconditional challenge distribution above.

A successful attack establishes an empirical lower bound on topic inference in this experiment. It does not establish a violation of Cloaked AI's underlying security notions. The [DCPE paper](https://eprint.iacr.org/2021/1666) studies notions including Real-or-Replaced and one-wayness under stated conditions; our game is a separate application-level topic question. Zero measured advantage does not prove topic privacy.

## Handoff to 2.2

Implemented in `phase2_export.py` and `phase2_views.py`: versioned allowlisted views, separate evaluator data, score/rank reproduction, private-field rejection, fresh event IDs, and isolated replay. Phase 1 file meanings and commands are preserved. `phase2_attack.py` now implements the fixed development attacks; `phase2_evaluate.py` freezes their outputs before reading evaluator labels. Fresh final queries remain a later milestone.
