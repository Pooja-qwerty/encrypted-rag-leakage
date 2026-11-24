# Project story

Encrypted vector search protects the stored representation while allowing useful retrieval. Our project asks what an observer can infer from the records returned by that search.

We separate two measurements:

- **Retrieval quality:** overlap with plaintext, plus precision@3 and nDCG@3 under synthetic relevance judgments.
- **Topic leakage:** query and session accuracy using only allowed logs and a disclosed known-label budget.

The current study compares plaintext and Cloaked retrieval on the same data and questions. At factor 1.5, retrieval relevance stays close to plaintext. With four known chunks per topic, direct voting averages about 75% query accuracy and 80% session accuracy from unordered IDs.

These results show topic leakage under the stated auxiliary knowledge. The attacker knows eight of the 24 chunk labels. With less prior knowledge, it is weaker, and clustering gives mixed benefits. The earlier near-chance results therefore do not establish topic privacy.

The cryptography contribution is to define the observables and auxiliary knowledge, keep the attacker within those inputs, and measure inference alongside retrieval quality. See the [threat model](01_THREAT_MODEL.md) and [results](02_RESULTS_SNAPSHOT.md).

The defense trace experiment adds dummy fetches without changing the delivered results. It weakens the tested attacks against a separate fetch-log observer, while the search server retains its original view. A full-download model keeps queries local at eight times the record-fetch cost. See the [defense results](04_DEFENSE_RESULTS.md).

The data and relevance judgments are synthetic. Larger corpus evaluation, independent label review, and the final report are still WIP.
