"""
Game v1 (Query Indistinguishability on a fixed DB): LeftIND-SL vs RightIND-SL.
Procedural (no classes) with game-level helpers so main() reads like playing the game.
"""

import time
import numpy as np


def key_gen(dim=64, seed=42):
    rng = np.random.default_rng(seed)
    q_mat, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    scale = float(rng.uniform(1.0, 5.0))
    return {"Q": q_mat, "scale": scale, "rng": rng}


def setup_corpus(key, n_chunks=500, dim=64, n_topics=8, spread=2.0):
    rng = key["rng"]
    centers = rng.normal(size=(n_topics, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    topics = rng.integers(0, n_topics, size=n_chunks)
    raw = centers[topics] + spread * rng.normal(size=(n_chunks, dim)) / np.sqrt(dim)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)

    enc_vectors = key["scale"] * (raw @ key["Q"].T)
    return {"vectors": enc_vectors, "topics": topics}


def embed(query_vec):
    v = np.asarray(query_vec, dtype=float)
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def encrypt_query(key, v_raw):
    return key["scale"] * (v_raw @ key["Q"].T)


def compute_similarity(db_vectors, q_enc):
    return db_vectors @ q_enc


def get_top_n(scores, top_n=5):
    k = min(top_n, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [int(i) for i in top_idx], [float(scores[i]) for i in top_idx]


def gen_log(db, key, query_vec, history, top_n=5):
    # 1. Encrypt Query Q on Client Side
    v_raw = embed(query_vec)
    q_k = encrypt_query(key, v_raw)

    # 2. Server-Side Similarity Search
    scores = compute_similarity(db["vectors"], q_k)

    # 3. Select Top-N Matching Chunks
    chunks_to_return, top_scores = get_top_n(scores, top_n=top_n)

    # 4. Generate Logs
    history.extend(chunks_to_return)
    return {
        "chunk_ids": chunks_to_return,
        "access_pattern": list(history),
        "timestamp": time.time(),
        "top_scores": top_scores,
    }


def initialize(seed=42):
    key = key_gen(seed=seed)
    db = setup_corpus(key)
    log = []
    history = []
    return key, db, log, history


def search_lr(side, key, db, log, history, q0, q1, top_n=5):
    # Left evaluates Q0; Right evaluates Q1
    chosen_q = q0 if side == "left" else q1
    entry = gen_log(db, key, chosen_q, history, top_n=top_n)
    log.append(entry)
    return entry


def finalize(log):
    return log


def adversary_guess(log, q0_expected_ids, q1_expected_ids):
    """Returns 1 if the returned chunk IDs look more like Q1's results than Q0's, else 0."""
    votes = 0
    for entry in log:
        returned = set(entry["chunk_ids"])
        votes += len(returned & q1_expected_ids) - len(returned & q0_expected_ids)
    return 1 if votes > 0 else 0


def play_game(side, q0, q1, steps, top_n, seed):
    """Runs one full LeftIND-SL or RightIND-SL game and returns the adversary's guess (0 or 1)."""
    ref_key, ref_db, _, _ = initialize(seed=seed)
    q0_ids = set(gen_log(ref_db, ref_key, q0, [], top_n=top_n)["chunk_ids"])
    q1_ids = set(gen_log(ref_db, ref_key, q1, [], top_n=top_n)["chunk_ids"])

    key, db, log, history = initialize(seed=seed)
    for _ in range(steps):
        search_lr(side, key, db, log, history, q0, q1, top_n=top_n)
    final_log = finalize(log)
    return adversary_guess(final_log, q0_ids, q1_ids)


def estimate_advantage(q0, q1, steps=5, top_n=5, trials=50, base_seed=1000):
    """Computes Adv_RAG^IND-SL(A) = Pr[A(Right IND-SL) -> 1] - Pr[A(Left IND-SL) -> 1]."""
    right_to_1 = 0
    left_to_1 = 0

    for t in range(trials):
        seed = base_seed + t
        left_to_1 += play_game("left", q0, q1, steps=steps, top_n=top_n, seed=seed)
        right_to_1 += play_game("right", q0, q1, steps=steps, top_n=top_n, seed=seed)

    pr_right_1 = right_to_1 / trials
    pr_left_1 = left_to_1 / trials
    return pr_right_1, pr_left_1, pr_right_1 - pr_left_1


def main():
    rng = np.random.default_rng(123)
    dim = 64

    # Adversary chooses challenge queries (Q0, Q1)
    q0 = rng.normal(size=dim)
    q1 = rng.normal(size=dim)

    # Play LeftIND-SL and RightIND-SL across trials and compute advantage
    pr_right_1, pr_left_1, adv = estimate_advantage(q0, q1, steps=5, top_n=5, trials=50)

    print(f"Pr[A(Right IND-SL) -> 1] = {pr_right_1:.3f}")
    print(f"Pr[A(Left  IND-SL) -> 1] = {pr_left_1:.3f}")
    print(f"Adv_RAG^IND-SL(A)        = {adv:.3f}")


if __name__ == "__main__":
    main()