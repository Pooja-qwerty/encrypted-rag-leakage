"""
Game v2 (Corpus Indistinguishability): Left_IND-SL vs Right_IND-SL.
Procedural (no classes) with multi-subject overlapping corpora, per-trial query jitter,
and a log-based adversary without an exact top-N oracle cheat sheet.
"""

import numpy as np


def make_10_subject_corpora(
    n_chunks=400,
    dim=64,
    n_subjects=10,
    overlap_ratio=0.85,
    focus_subject=1,
    spread=0.9,
    chunk_size=256,
    seed=42,
):
    """
    Creates two matching-size 10-subject corpora (db0, db1) sharing overlap_ratio
    of chunks, with the remaining chunks differing around focus_subject.
    """
    rng = np.random.default_rng(seed)

    centers = rng.normal(size=(n_subjects, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    # Secondary sub-topic direction within focus_subject
    subtopic_dir = rng.normal(size=dim)
    subtopic_dir -= (subtopic_dir @ centers[focus_subject]) * centers[focus_subject]
    subtopic_dir /= np.linalg.norm(subtopic_dir)

    topics = np.arange(n_chunks) % n_subjects
    base_vectors = np.zeros((n_chunks, dim), dtype=float)
    for i in range(n_chunks):
        c = centers[topics[i]]
        v = c + spread * rng.normal(size=dim) / np.sqrt(dim)
        base_vectors[i] = v / np.linalg.norm(v)

    db0_vectors = base_vectors.copy()
    db1_vectors = base_vectors.copy()

    focus_indices = [i for i in range(n_chunks) if topics[i] == focus_subject]
    n_shared_in_focus = int(round(len(focus_indices) * overlap_ratio))
    modified_indices = focus_indices[n_shared_in_focus:]

    # DB1 aligns slightly closer to the sub-topic; DB0 shifts slightly away,
    # keeping overlap so neither side is 0% or 100%.
    for idx in modified_indices:
        noise = (spread * 0.85) * rng.normal(size=dim) / np.sqrt(dim)
        v1 = centers[focus_subject] + 0.35 * subtopic_dir + noise
        db1_vectors[idx] = v1 / np.linalg.norm(v1)

        v0 = centers[focus_subject] - 0.25 * subtopic_dir + noise
        db0_vectors[idx] = v0 / np.linalg.norm(v0)

    chunk_sizes = [int(chunk_size)] * n_chunks
    focus_chunk_ids = set(int(i) for i in modified_indices)

    db0 = {"vectors": db0_vectors, "topics": topics, "chunk_sizes": list(chunk_sizes)}
    db1 = {"vectors": db1_vectors, "topics": topics, "chunk_sizes": list(chunk_sizes)}
    aux_info = {
        "centers": centers,
        "subtopic_dir": subtopic_dir,
        "topics": topics,
        "focus_subject": focus_subject,
        "focus_chunk_ids": focus_chunk_ids,
    }
    return db0, db1, aux_info


def make_query_stream(
    centers,
    subtopic_dir,
    focus_subject=1,
    n_queries=16,
    repeat_prob=0.30,
    jitter=0.55,
    seed=100,
):
    """
    Generates a stream of queries across the 10 subjects with per-trial jitter
    and occasional exact repeats so SP / AP+SP has signal.
    """
    rng = np.random.default_rng(seed)
    n_subjects = len(centers)
    stream = []

    for _ in range(n_queries):
        if stream and rng.random() < repeat_prob:
            prev_idx = int(rng.integers(0, len(stream)))
            stream.append(stream[prev_idx].copy())
            continue

        if rng.random() < 0.45:
            base = centers[focus_subject] + 0.25 * subtopic_dir
        else:
            subj = int(rng.integers(0, n_subjects))
            base = centers[subj]

        q = base + jitter * rng.normal(size=centers.shape[1]) / np.sqrt(centers.shape[1])
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        stream.append(q)

    return stream


def key_gen(seed=42):
    """Stand-in key generator (unused by Leak under hypothetical leakage profile L)."""
    rng = np.random.default_rng(seed)
    return {"key_id": int(rng.integers(10**6, 10**9))}


def setup_corpus(raw_db):
    """Prepares corpus for server-side similarity search."""
    return {
        "vectors": np.asarray(raw_db["vectors"], dtype=float),
        "topics": np.asarray(raw_db["topics"], dtype=int),
        "chunk_sizes": list(raw_db["chunk_sizes"]),
    }


def top_n(db, query_vec, n_results=5):
    """Returns indices of top-N closest/most-similar chunks in descending score order."""
    q = np.asarray(query_vec, dtype=float)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    scores = db["vectors"] @ q
    k = min(n_results, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [int(i) for i in idx]


def query_key(query_vec):
    """Canonical hashable key for repeat-token tracking in AP+SP without leaking raw Q."""
    arr = np.round(np.asarray(query_vec, dtype=float), decimals=8)
    return arr.tobytes()


def leak(db, seen, query_vec, n_results=5, level="AP-set"):
    """Enforces leakage profile L: outputs only what L permits."""
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
        qk = query_key(query_vec)
        if qk in seen:
            token = seen[qk]
        else:
            token = len(seen) + 1
            seen[qk] = token
        entry["repeat_token"] = token
        return entry

    raise ValueError(f"Unknown leakage level: {level}")


def initialize(db0, db1, side, seed=42):
    """
    Initialize(DB0, DB1):
    - Aborts unless DB0 and DB1 have the same n and same chunk sizes
    - Generates K, sets up DB0 (Left) or DB1 (Right), initializes Log and Seen
    - Returns (state, l_setup)
    """
    n0 = len(db0["vectors"])
    n1 = len(db1["vectors"])
    if n0 != n1 or db0["chunk_sizes"] != db1["chunk_sizes"]:
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
    l_setup = {"n": n0, "chunk_sizes": list(db0["chunk_sizes"])}
    return state, l_setup


def search(state, query_vec, n_results=5, level="AP-set"):
    """Search(Q): evaluates the same query Q in both worlds through Leak()."""
    entry = leak(
        db=state["db"],
        seen=state["seen"],
        query_vec=query_vec,
        n_results=n_results,
        level=level,
    )
    state["log"].append(entry)
    return entry


def finalize(d):
    """Finalize(d): returns the adversary's output bit d."""
    return d


def adversary_guess(l_setup, log, focus_chunk_ids):
    """
    Log-based adversary (no exact per-query oracle from db0 or db1):
    Uses chunk hit counts, rank-order weight (AP-ordered), and repeat-token
    confirmation (AP+SP) to decide if the log came from DB1 (1) or DB0 (0).
    """
    if not log or "chunk_ids" not in log[0]:
        return 0

    token_counts = {}
    for entry in log:
        tok = entry.get("repeat_token")
        if tok is not None:
            token_counts[tok] = token_counts.get(tok, 0) + 1

    score = 0.0
    for entry in log:
        obs = entry["chunk_ids"]
        tok = entry.get("repeat_token")

        if isinstance(obs, list):
            # Rank-sensitive score: top ranks carry stronger signal
            step_score = sum(
                1.0 / (1.0 + rank)
                for rank, cid in enumerate(obs)
                if cid in focus_chunk_ids
            )
            # In AP+SP, repeated queries confirm non-accidental hits
            if tok is not None and token_counts.get(tok, 1) > 1 and step_score > 0:
                step_score *= 1.35
            score += step_score
        else:
            # Unordered set hit count
            hits = sum(1 for cid in obs if cid in focus_chunk_ids)
            score += 0.45 * hits

    return 1 if score >= 3.2 else 0


def play_corpus_game(side, db0, db1, query_stream, focus_chunk_ids, n_results=5, level="AP-set"):
    """Plays one Left_IND-SL or Right_IND-SL game and returns adversary bit d in {0, 1}."""
    state, _l_setup = initialize(db0, db1, side=side)
    for q in query_stream:
        search(state, q, n_results=n_results, level=level)

    d = adversary_guess(_l_setup, state["log"], focus_chunk_ids)
    return finalize(d)


def estimate_advantage(db0, db1, aux_info, n_results=5, level="AP-set", trials=200, base_seed=500):
    """Computes Adv_RAG^IND-SL(A) = Pr[A(Right) -> 1] - Pr[A(Left) -> 1] across jittered streams."""
    right_to_1 = 0
    left_to_1 = 0

    for t in range(trials):
        q_stream = make_query_stream(
            centers=aux_info["centers"],
            subtopic_dir=aux_info["subtopic_dir"],
            focus_subject=aux_info["focus_subject"],
            n_queries=16,
            repeat_prob=0.30,
            jitter=0.55,
            seed=base_seed + t,
        )
        if play_corpus_game("left", db0, db1, q_stream, aux_info["focus_chunk_ids"], n_results, level) == 1:
            left_to_1 += 1
        if play_corpus_game("right", db0, db1, q_stream, aux_info["focus_chunk_ids"], n_results, level) == 1:
            right_to_1 += 1

    pr_right_1 = right_to_1 / trials
    pr_left_1 = left_to_1 / trials
    return pr_right_1, pr_left_1, pr_right_1 - pr_left_1


def main():
    db0, db1, aux_info = make_10_subject_corpora(
        n_chunks=400,
        dim=64,
        n_subjects=10,
        overlap_ratio=0.80,
        focus_subject=1,
        spread=0.9,
        chunk_size=256,
        seed=42,
    )

    for level in ["Volume", "AP-set", "AP-ordered", "AP+SP"]:
        pr_r, pr_l, adv = estimate_advantage(
            db0, db1, aux_info, n_results=5, level=level, trials=200, base_seed=500
        )
        print(f"[{level:10s}] Pr(Right->1)={pr_r:.3f}  Pr(Left->1)={pr_l:.3f}  Adv={adv:.3f}")


if __name__ == "__main__":
    main()