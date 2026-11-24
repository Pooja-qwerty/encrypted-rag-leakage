"""
Game IND-SL (Corpus Indistinguishability on Real Text): Left_IND-SL vs Right_IND-SL.
Procedural (no classes) using data/CS.txt and data/BIO.txt.

Adversary model note (see build_reference_signals / adversary_guess):
In Game IND-SL the adversary chooses DB0 and DB1 in Initialize(), so it is
assumed to know both corpora and the query distribution. The reference-signal
precomputation below models that prior knowledge; it is NOT leakage from the
server. The server-side leak is exactly Leak(...).
"""

from pathlib import Path
import math
import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    HAS_ST = True
except ImportError:
    HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer


VALID_LEVELS = ("Volume", "AP-set", "AP-ordered", "AP+SP")
# Information-theoretic ordering: leakage grows down this list.
LEVEL_ORDER = ["Volume", "AP-set", "AP-ordered", "AP+SP"]


def load_subject_chunks(cs_path="data/CS.txt", bio_path="data/BIO.txt"):
    """Loads CS and Bio chunks split by blank lines from disk."""
    base_dir = Path(__file__).resolve().parent
    cs_file = Path(cs_path) if Path(cs_path).is_absolute() else base_dir / cs_path
    bio_file = Path(bio_path) if Path(bio_path).is_absolute() else base_dir / bio_path

    cs_texts = [b.strip() for b in cs_file.read_text(encoding="utf-8").strip().split("\n\n") if b.strip()]
    bio_texts = [b.strip() for b in bio_file.read_text(encoding="utf-8").strip().split("\n\n") if b.strip()]
    return cs_texts, bio_texts


def build_embedder(corpus_texts, model_name="all-MiniLM-L6-v2"):
    """
    Returns an embed_fn(list_of_strings) -> normalized (M, D) float64 array.
    Uses SentenceTransformer if available; falls back to TF-IDF vectors otherwise.
    """
    if HAS_ST:
        try:
            model = SentenceTransformer(model_name)

            def st_embed(queries):
                vecs = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
                return np.asarray(vecs, dtype=float)

            return st_embed
        except Exception:
            pass

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    vectorizer.fit(corpus_texts)

    def tfidf_embed(queries):
        mat = vectorizer.transform(queries).toarray().astype(float)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    return tfidf_embed


def make_cs_bio_corpora(cs_texts, bio_texts, embed_fn, overlap_ratio=0.75, chunk_size=256):
    """
    Creates two equal-length corpora (db0, db1) from CS.txt and BIO.txt:
    - Both share `overlap_ratio` of chunks identically in the first `n_shared` slots.
    - The remaining `n_diff` slots differ: DB0 holds CS chunks, DB1 holds Bio chunks,
      with identical chunk_sizes so Initialize(DB0, DB1) does not abort.
    """
    n_slots = min(len(cs_texts), len(bio_texts))
    if n_slots < 4:
        raise ValueError("Need at least 4 chunks in both CS.txt and BIO.txt.")

    n_shared = max(1, min(n_slots - 1, int(round(n_slots * overlap_ratio))))
    n_diff = n_slots - n_shared

    shared_texts = []
    shared_topics = []
    for i in range(n_shared):
        if i % 2 == 0:
            shared_texts.append(cs_texts[i])
            shared_topics.append("CS")
        else:
            shared_texts.append(bio_texts[i])
            shared_topics.append("Bio")

    db0_diff_texts = [cs_texts[n_shared + j] for j in range(n_diff)]
    db1_diff_texts = [bio_texts[n_shared + j] for j in range(n_diff)]

    db0_texts = shared_texts + db0_diff_texts
    db1_texts = shared_texts + db1_diff_texts

    db0_topics = shared_topics + ["CS"] * n_diff
    db1_topics = shared_topics + ["Bio"] * n_diff

    db0_vectors = embed_fn(db0_texts)
    db1_vectors = embed_fn(db1_texts)

    # Setup leakage sees n and size of each encrypted chunk; must match across DB0 and DB1
    chunk_sizes = [int(chunk_size)] * n_slots
    diff_slot_ids = set(range(n_shared, n_slots))

    db0 = {
        "texts": db0_texts,
        "vectors": db0_vectors,
        "topics": db0_topics,
        "chunk_sizes": list(chunk_sizes),
    }
    db1 = {
        "texts": db1_texts,
        "vectors": db1_vectors,
        "topics": db1_topics,
        "chunk_sizes": list(chunk_sizes),
    }
    aux_info = {
        "diff_slot_ids": diff_slot_ids,
        "n_shared": n_shared,
        "n_diff": n_diff,
    }
    return db0, db1, aux_info


def get_candidate_queries():
    """Queries spanning CS and Bio concepts that overlap in retrieval across slots."""
    return [
        "How does indexing speed up search and lookup?",
        "Explain sequence alignment and similarity scoring.",
        "How do tree structures organize hierarchical data or evolutionary branching?",
        "How are graphs used to traverse networks and pathways?",
        "What is sequencing and how is order reconstructed?",
        "How does k-mer or inverted indexing improve pattern matching?",
        "Explain how genome and database indexing map substrings or keys to positions.",
        "How does dynamic programming solve sequence comparison?",
        "What methods are used for fast motif and text search?",
        "How do read alignment and query optimization reduce search cost?",
    ]


def make_query_stream(
    query_pool_texts,
    query_pool_vecs,
    n_queries=12,
    repeat_prob=0.30,
    jitter=0.12,
    seed=100,
):
    """
    Generates a stream of (query_text, query_vec) pairs with occasional exact
    repeats (for AP+SP repeat tokens) and per-trial vector jitter for partial
    separation.

    A "repeat" reuses the *identical* (text, vector) pair from an earlier query,
    so query_key() in Leak maps it to the same repeat_token, as intended by AP+SP.
    """
    rng = np.random.default_rng(seed)
    n_pool = len(query_pool_texts)
    dim = query_pool_vecs.shape[1]
    stream = []

    for _ in range(n_queries):
        if stream and rng.random() < repeat_prob:
            prev_idx = int(rng.integers(0, len(stream)))
            q_txt, q_vec = stream[prev_idx]
            stream.append((q_txt, q_vec.copy()))
            continue

        idx = int(rng.integers(0, n_pool))
        q_txt = query_pool_texts[idx]
        base = query_pool_vecs[idx]

        q = base + jitter * rng.normal(size=dim) / np.sqrt(dim)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        stream.append((q_txt, q))

    return stream


def key_gen(seed=42):
    """Generates key K (unused by Leak under leakage profile L)."""
    rng = np.random.default_rng(seed)
    return {"key_id": int(rng.integers(10**6, 10**9))}


def setup_corpus(raw_db):
    """Prepares corpus for server-side search."""
    return {
        "texts": list(raw_db["texts"]),
        "vectors": np.asarray(raw_db["vectors"], dtype=float),
        "topics": list(raw_db["topics"]),
        "chunk_sizes": list(raw_db["chunk_sizes"]),
    }


def top_n(db, query_vec, n_results=3):
    """Returns indices of top-N most similar chunks in descending score order."""
    assert n_results >= 1, "n_results must be >= 1"
    q = np.asarray(query_vec, dtype=float)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    scores = db["vectors"] @ q
    k = min(n_results, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [int(i) for i in idx]


def query_key(query_text, query_vec):
    """Canonical query key for repeat-token tracking in AP+SP."""
    arr = np.round(np.asarray(query_vec, dtype=float), decimals=8)
    return query_text.encode("utf-8") + b"|" + arr.tobytes()


def leak(db, seen, query_text, query_vec, n_results=3, level="AP-set"):
    """
    Leak(DB, Seen, Q): stands in for server-side search over encrypted data
    and outputs only what leakage profile L allows.
    """
    if level not in VALID_LEVELS:
        raise ValueError(f"Unknown leakage level: {level}")

    r = top_n(db, query_vec, n_results=n_results)
    entry = {"num_results": len(r)}

    if level == "Volume":
        return entry

    if level == "AP-set":
        entry["chunk_ids"] = set(r)
        return entry

    if level == "AP-ordered":
        entry["chunk_ids"] = list(r)
        return entry

    if level == "AP+SP":
        entry["chunk_ids"] = list(r)
        qk = query_key(query_text, query_vec)
        if qk in seen:
            token = seen[qk]
        else:
            token = len(seen) + 1
            seen[qk] = token
        entry["repeat_token"] = token
        return entry


def initialize(db0, db1, side, seed=42):
    """
    Initialize(DB0, DB1):
    - Aborts unless DB0 and DB1 have the same n and same chunk sizes
    - K <- KeyGen(), DB <- SetupCorpus(DB0 or DB1), Log <- [], Seen <- {}
    - Returns (state, l_setup) where l_setup = {"n": n, "chunk_sizes": ...}
    """
    n0 = len(db0["vectors"])
    n1 = len(db1["vectors"])
    if n0 != n1 or list(db0["chunk_sizes"]) != list(db1["chunk_sizes"]):
        raise ValueError("Abort: DB0 and DB1 must have the same n and same chunk sizes.")

    _k = key_gen(seed=seed)
    chosen_raw_db = db0 if side == "left" else db1
    db = setup_corpus(chosen_raw_db)

    state = {
        "side": side,
        "db": db,
        "log": [],
        "seen": {},
    }
    l_setup = {
        "n": n0,
        "chunk_sizes": list(db0["chunk_sizes"]),
    }
    return state, l_setup


def search(state, query_text, query_vec, n_results=3, level="AP-set"):
    """
    Search(Q): evaluates the same query Q in both worlds via Leak(),
    appends to Log, and returns L.
    """
    entry = leak(
        db=state["db"],
        seen=state["seen"],
        query_text=query_text,
        query_vec=query_vec,
        n_results=n_results,
        level=level,
    )
    state["log"].append(entry)
    return entry


def finalize(d):
    """Finalize(d): returns adversary bit d in {0, 1}."""
    return int(d)


def build_reference_signals(db0, db1, query_pool_vecs, diff_slot_ids,
                            n_results=3, n_samples=400, jitter=0.12, seed=123):
    """
    OFFLINE ADVERSARY PREPROCESSING.

    Computes per-slot signed, normalized evidence that a slot is favored by
    DB1 (+) vs DB0 (-) by sampling queries from the query pool and running
    TopN against *both* corpora.

    Justification: in Game IND-SL the adversary chooses DB0 and DB1 during
    Initialize(), so it is assumed to know both corpora. It also knows the
    query distribution (the pool). This auxiliary signal therefore models
    the adversary's prior knowledge, consistent with the game. It is NOT
    server leakage; the server-side leak remains exactly Leak(...).

    Rank-aware and calibrated:
      - Rank weights: rank 0 is stronger evidence than rank n_results-1.
      - Signed: +1 = maximally DB1-favored, -1 = maximally DB0-favored.
      - Normalized so a strongly DB1-favored slot and a strongly DB0-favored
        slot contribute equal magnitude. This removes DB1/DB0 asymmetry.

    Returns: dict {cid: signed_strength in [-1, +1]} for cid in diff_slot_ids.
    """
    rng = np.random.default_rng(seed)
    n_pool = len(query_pool_vecs)
    dim = query_pool_vecs.shape[1]

    db1_rank = {cid: [0] * n_results for cid in diff_slot_ids}
    db0_rank = {cid: [0] * n_results for cid in diff_slot_ids}

    for _ in range(n_samples):
        idx = int(rng.integers(0, n_pool))
        q = query_pool_vecs[idx] + jitter * rng.normal(size=dim) / np.sqrt(dim)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        for rank, cid in enumerate(top_n(db1, q, n_results=n_results)):
            if cid in diff_slot_ids:
                db1_rank[cid][rank] += 1
        for rank, cid in enumerate(top_n(db0, q, n_results=n_results)):
            if cid in diff_slot_ids:
                db0_rank[cid][rank] += 1

    rank_weights = [1.0 / (1 + r) for r in range(n_results)]
    db1_score = {cid: sum(w * c for c, w in zip(db1_rank[cid], rank_weights)) for cid in diff_slot_ids}
    db0_score = {cid: sum(w * c for c, w in zip(db0_rank[cid], rank_weights)) for cid in diff_slot_ids}

    slot_strength = {}
    for cid in diff_slot_ids:
        denom = db1_score[cid] + db0_score[cid]
        if denom <= 0:
            slot_strength[cid] = 0.0
        else:
            slot_strength[cid] = (db1_score[cid] - db0_score[cid]) / denom
    return slot_strength

def adversary_guess(l_setup, log, slot_strength, level="AP-set"):
    """
    Adversary strategy: consumes only (l_setup, log) from the game, plus the
    offline-derived per-slot signed evidence (justified by the game giving
    the adversary both DB0 and DB1; see build_reference_signals).

    Decision rule: output DB1 iff the sum of signed evidence over observed
    slots is > 0. Signed evidence is symmetric, so there is no built-in DB1
    bias; ties default to DB0.

    Returns 1 if evidence favors DB1 (Right world), else 0.
    """
    if level == "Volume" or not log or "chunk_ids" not in log[0]:
        return 0

    token_counts = {}
    if level == "AP+SP":
        for entry in log:
            tok = entry.get("repeat_token")
            if tok is not None:
                token_counts[tok] = token_counts.get(tok, 0) + 1

    score = 0.0
    for entry in log:
        obs = entry["chunk_ids"]
        tok = entry.get("repeat_token")
        repeat_boost = 1.25 if (level == "AP+SP" and tok is not None
                                and token_counts.get(tok, 1) > 1) else 1.0

        if isinstance(obs, list) and level in ("AP-ordered", "AP+SP"):
            for rank, cid in enumerate(obs):
                if cid in slot_strength:
                    score += repeat_boost * slot_strength[cid] / (1.0 + rank)
        else:
            for cid in obs:
                if cid in slot_strength:
                    score += 0.5 * repeat_boost * slot_strength[cid]

    return 1 if score > 0.0 else 0

def play_corpus_game(side, db0, db1, query_stream, slot_strength,
                     n_results=3, level="AP-set", seed=42):
    """Runs one Left_IND-SL (DB0) or Right_IND-SL (DB1) game and returns bit d."""
    state, l_setup = initialize(db0, db1, side=side, seed=seed)
    for q_txt, q_vec in query_stream:
        search(state, q_txt, q_vec, n_results=n_results, level=level)

    d = adversary_guess(l_setup, state["log"], slot_strength, level=level)
    return finalize(d)


def estimate_advantage(
    db0,
    db1,
    aux_info,
    query_pool_texts,
    query_pool_vecs,
    n_results=3,
    level="AP-set",
    trials=500,
    base_seed=500,
    jitter=0.12,
):
    """
    Estimates Adv_RAG^IND-SL(A) = Pr[A(Right) -> 1] - Pr[A(Left) -> 1].
    Returns (pr_right_1, pr_left_1, adv, (ci_lo, ci_hi)) with a 95% CI on adv.
    """
    slot_strength = build_reference_signals(
        db0=db0,
        db1=db1,
        query_pool_vecs=query_pool_vecs,
        diff_slot_ids=aux_info["diff_slot_ids"],
        n_results=n_results,
        jitter=jitter,
    )

    right_to_1 = 0
    left_to_1 = 0

    for t in range(trials):
        q_stream = make_query_stream(
            query_pool_texts=query_pool_texts,
            query_pool_vecs=query_pool_vecs,
            n_queries=12,
            repeat_prob=0.30,
            jitter=jitter,
            seed=base_seed + t,
        )
        if play_corpus_game("left", db0, db1, q_stream, slot_strength,
                            n_results, level, seed=base_seed + t) == 1:
            left_to_1 += 1
        if play_corpus_game("right", db0, db1, q_stream, slot_strength,
                            n_results, level, seed=base_seed + t) == 1:
            right_to_1 += 1

    pr_right_1 = right_to_1 / trials
    pr_left_1 = left_to_1 / trials
    adv = pr_right_1 - pr_left_1
    se = math.sqrt(pr_right_1 * (1 - pr_right_1) / trials
                   + pr_left_1 * (1 - pr_left_1) / trials)
    ci = (adv - 1.96 * se, adv + 1.96 * se)
    return pr_right_1, pr_left_1, adv, ci

def main():
    cs_texts, bio_texts = load_subject_chunks("data/CS.txt", "data/BIO.txt")
    embed_fn = build_embedder(cs_texts + bio_texts)

    db0, db1, aux_info = make_cs_bio_corpora(
        cs_texts=cs_texts,
        bio_texts=bio_texts,
        embed_fn=embed_fn,
        overlap_ratio=0.50,
        chunk_size=256,
    )

    query_pool_texts = get_candidate_queries()
    query_pool_vecs = embed_fn(query_pool_texts)

    n_slots = len(db0["texts"])
    print(f"Loaded {n_slots} slots per corpus "
          f"({aux_info['n_shared']} shared CS/Bio, {aux_info['n_diff']} differing slots)")
    print(f"DB0 differing slots: CS | DB1 differing slots: Bio | "
          f"Differing IDs: {sorted(aux_info['diff_slot_ids'])}\n")

    print(f"{'Leakage':10s} | {'Right->1':9s} | {'Left->1':8s} | "
          f"{'Advantage':9s} | {'95% CI':18s} | {'Signal'}")
    print("-" * 88)

    advs = {}
    for level in VALID_LEVELS:
        pr_r, pr_l, adv, ci = estimate_advantage(
            db0, db1, aux_info, query_pool_texts, query_pool_vecs,
            n_results=3, level=level, trials=2000, base_seed=500, jitter=0.06,
        )
        advs[level] = adv

        if abs(adv) <= 0.01:
            signal = "No separation"
        elif ci[0] > 0 or ci[1] < 0:
            signal = "Significant (CI excludes 0)"
        else:
            signal = "Trend, not significant"

        ci_str = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        print(f"{level:10s} | {pr_r:9.3f} | {pr_l:8.3f} | "
              f"{adv:+9.3f} | {ci_str:18s} | {signal}")

    # --- Sanity checks ------------------------------------------------------
    assert abs(advs["Volume"]) < 0.01, \
        f"Volume advantage must be ~0, got {advs['Volume']:+.3f}"

    print()
    for a, b in zip(LEVEL_ORDER, LEVEL_ORDER[1:]):
        if advs[b] + 0.05 < advs[a]:
            print(f"WARNING: {b} ({advs[b]:+.3f}) < {a} ({advs[a]:+.3f}) by >0.05 — "
                  f"violates information monotonicity; likely noise or "
                  f"suboptimal adversary strategy.")


if __name__ == "__main__":
    main()