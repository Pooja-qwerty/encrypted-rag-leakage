"""
Game IND-QT-PS: plaintext baseline with extra attacker knowledge.
Calibration uses private plaintext embeddings and a public corpus.
For the restricted log-only attack, use log_attack.py.
"""

from pathlib import Path
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Data / embeddings / DB setup
# ---------------------------------------------------------------------------

def load_file(path):
    """Returns list of chunk strings from one file, split by blank lines."""
    base_dir = Path(__file__).resolve().parent
    f = Path(path) if Path(path).is_absolute() else base_dir / path
    raw = f.read_text(encoding="utf-8").strip()
    return [b.strip() for b in raw.split("\n\n") if b.strip()]


def build_embedder(corpus_texts, model_name="all-MiniLM-L6-v2"):
    if HAS_ST:
        try:
            model = SentenceTransformer(model_name)
            def st_embed(qs):
                v = model.encode(qs, convert_to_numpy=True, normalize_embeddings=True)
                return np.asarray(v, dtype=float)
            return st_embed
        except Exception:
            pass
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    vec.fit(corpus_texts)
    def tfidf_embed(qs):
        m = vec.transform(qs).toarray().astype(float)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n
    return tfidf_embed


def setup_db(raw_corpus, embed_fn):
    return {
        "texts": list(raw_corpus["texts"]),
        "subjects": list(raw_corpus["subjects"]),
        "chunk_sizes": list(raw_corpus["chunk_sizes"]),
        "vectors": embed_fn(raw_corpus["texts"]),
    }


def top_n(db, query_vec, n_results=3):
    q = np.asarray(query_vec, dtype=float)
    n = np.linalg.norm(q)
    if n > 0:
        q = q / n
    scores = db["vectors"] @ q
    k = min(n_results, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [int(i) for i in idx]


def query_key(query_text):
    return query_text.encode("utf-8")


def leak(db, seen, query_text, query_vec, n_results=3, level="AP-set"):
    """Leakage profile L. Only what the server sees."""
    r = top_n(db, query_vec, n_results=n_results)
    entry = {"num_results": len(r)}
    if level == "Volume":
        return entry
    if level == "AP-set":
        entry["chunk_ids"] = sorted(set(r))
        return entry
    if level == "AP-ordered":
        entry["chunk_ids"] = list(r)
        return entry
    if level == "AP+SP":
        entry["chunk_ids"] = list(r)
        qk = query_key(query_text)
        if qk in seen:
            token = seen[qk]
        else:
            token = len(seen) + 1
            seen[qk] = token
        entry["repeat_token"] = token
        return entry
    raise ValueError(f"Unknown level: {level}")


def build_public_corpus(public_raw, embed_fn):
    """Adversary's auxiliary input: unencrypted, labeled public corpus."""
    return {
        "texts": list(public_raw["texts"]),
        "subjects": list(public_raw["subjects"]),
        "vectors": embed_fn(public_raw["texts"]),
    }


# ---------------------------------------------------------------------------
# Adversary: PublicCalibrate + Classify (passive, no probing)
# ---------------------------------------------------------------------------

def public_calibrate(db, public_corpus, noise_std=0.0, seed=0):
    """
    Extra-knowledge baseline: compare private plaintext embeddings to public
    embeddings. `noise_std` perturbs that private information. This is not
    available to the restricted log-only attacker.
    """
    rng = np.random.default_rng(seed)
    priv_vecs = np.asarray(db["vectors"], dtype=float)
    if noise_std > 0:
        priv_vecs = priv_vecs + noise_std * rng.normal(size=priv_vecs.shape)
        n = np.linalg.norm(priv_vecs, axis=1, keepdims=True)
        n[n == 0] = 1.0
        priv_vecs = priv_vecs / n

    pub_vecs = public_corpus["vectors"]
    pub_topics = public_corpus["subjects"]

    chunk_map = {}
    for cid in range(len(priv_vecs)):
        sims = pub_vecs @ priv_vecs[cid]
        nearest = int(np.argmax(sims))
        chunk_map[cid] = pub_topics[nearest]
    return chunk_map


def classify(log, chunk_map, level):
    """Adversary's guess from log + chunk-topic map. 1=Bio(Right), 0=CS(Left)."""
    if level == "Volume" or not log or "chunk_ids" not in log[0]:
        return 0

    score_cs, score_bio = 0.0, 0.0
    for entry in log:
        obs = entry["chunk_ids"]
        if isinstance(obs, list) and level in ("AP-ordered", "AP+SP"):
            for rank, cid in enumerate(obs):
                if cid in chunk_map:
                    w = 1.0 / (1.0 + rank)
                    if chunk_map[cid] == "CS":
                        score_cs += w
                    else:
                        score_bio += w
        else:
            for cid in obs:
                if cid in chunk_map:
                    if chunk_map[cid] == "CS":
                        score_cs += 1.0
                    else:
                        score_bio += 1.0
    return 1 if score_bio > score_cs else 0


# ---------------------------------------------------------------------------
# Game: Initialize / Challenge / Finalize
# ---------------------------------------------------------------------------

def initialize_qt_ps(db, public_corpus, side, noise_std=0.0, seed=0):
    state = {
        "side": side,
        "db": db,
        "log": [],
        "seen": {},
        "map": public_calibrate(db, public_corpus,
                                noise_std=noise_std, seed=seed),
    }
    l_setup = {"n": len(db["texts"]), "chunk_sizes": list(db["chunk_sizes"])}
    return state, l_setup


def sample_query(pool, q_vecs, rng):
    q = pool[int(rng.integers(0, len(pool)))]
    return q, q_vecs[q]


def play_qt_ps_trial(db, public_corpus, cs_pool, bio_pool, q_vecs,
                     n_results=3, level="AP-set", noise_std=0.0,
                     n_repeats=1, seed=0):
    """
    One Left trial and one Right trial with freshly sampled queries.
    Repeats give identical votes; this classifier does not use repeat tokens.
    Map is computed per-trial using this trial's seed (averages over map noise).
    """
    rng = np.random.default_rng(seed)

    left_state, _ = initialize_qt_ps(db, public_corpus, "left",
                                     noise_std=noise_std, seed=seed)
    right_state, _ = initialize_qt_ps(db, public_corpus, "right",
                                      noise_std=noise_std, seed=seed)

    q0_text, q0_vec = sample_query(cs_pool, q_vecs, rng)
    q1_text, q1_vec = sample_query(bio_pool, q_vecs, rng)

    for _ in range(n_repeats):
        l_left = leak(left_state["db"], left_state["seen"],
                      q0_text, q0_vec, n_results, level)
        l_right = leak(right_state["db"], right_state["seen"],
                       q1_text, q1_vec, n_results, level)
        left_state["log"].append(l_left)
        right_state["log"].append(l_right)

    d_left = classify(left_state["log"], left_state["map"], level)
    d_right = classify(right_state["log"], right_state["map"], level)
    return d_right, d_left


def estimate_qt_ps_advantage(db, public_corpus, cs_pool, bio_pool, q_vecs,
                              n_results=3, level="AP-set", noise_std=0.0,
                              n_repeats=1, trials=200, seed=42):
    right_to_1 = 0
    left_to_1 = 0
    for t in range(trials):
        d_r, d_l = play_qt_ps_trial(
            db, public_corpus, cs_pool, bio_pool, q_vecs,
            n_results, level, noise_std, n_repeats, seed=seed + t,
        )
        right_to_1 += d_r
        left_to_1 += d_l
    pr_r = right_to_1 / trials
    pr_l = left_to_1 / trials
    return pr_r, pr_l, pr_r - pr_l


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("Extra-knowledge plaintext baseline: calibration reads private embeddings.\n")
    cs_priv = load_file("data/CS.txt")
    bio_priv = load_file("data/BIO.txt")
    cs_pub  = load_file("data/CS_public.txt")
    bio_pub = load_file("data/BIO_public.txt")

    private_raw = {
        "texts": cs_priv + bio_priv,
        "subjects": ["CS"] * len(cs_priv) + ["Bio"] * len(bio_priv),
        "chunk_sizes": [len(t.encode("utf-8")) for t in cs_priv + bio_priv],
    }
    public_raw = {
        "texts": cs_pub + bio_pub,
        "subjects": ["CS"] * len(cs_pub) + ["Bio"] * len(bio_pub),
        "chunk_sizes": [len(t.encode("utf-8")) for t in cs_pub + bio_pub],
    }

    cs_pool = [
        "How does indexing speed up database search?",
        "What is a B-tree used for?",
        "Explain hash tables in computer science.",
        "How do graph traversal algorithms work?",
    ]
    bio_pool = [
        "How does genome sequencing work?",
        "What is sequence alignment in bioinformatics?",
        "Explain phylogenetic trees and evolution.",
        "How are protein structures predicted?",
    ]

    all_texts = private_raw["texts"] + public_raw["texts"] + cs_pool + bio_pool
    embed_fn = build_embedder(all_texts)

    db = setup_db(private_raw, embed_fn)
    public_corpus = build_public_corpus(public_raw, embed_fn)

    all_queries = cs_pool + bio_pool
    q_vecs_arr = embed_fn(all_queries)
    q_vecs = {q: q_vecs_arr[i] for i, q in enumerate(all_queries)}

    print(f"Private DB: {len(db['texts'])} chunks "
          f"({db['subjects'].count('CS')} CS, {db['subjects'].count('Bio')} Bio)")
    print(f"Public corpus: {len(public_corpus['texts'])} chunks "
          f"({public_corpus['subjects'].count('CS')} CS, "
          f"{public_corpus['subjects'].count('Bio')} Bio)")
    print(f"Query pool: {len(cs_pool)} CS, {len(bio_pool)} Bio\n")

    # ---- Diagnostic: average map balance over many seeds ----
    from collections import Counter
    fracs = []
    for s in range(50):
        st, _ = initialize_qt_ps(db, public_corpus, "left",
                                 noise_std=0.00, seed=s)
        c = Counter(st["map"].values())
        fracs.append(c["CS"] / (c["CS"] + c["Bio"]))
    print(f"Map frac CS over 50 seeds: mean={np.mean(fracs):.2f}, "
          f"min={min(fracs):.2f}, max={max(fracs):.2f}")
    if not (0.25 <= np.mean(fracs) <= 0.75):
        print("WARNING: map biased on average — results may be inflated.")
    print()

    for noise_std in [0.00, 0.05]:
        print(f"\n--- noise_std = {noise_std} ---")
        print(f"{'Leakage':10s} | {'Right->1':9s} | {'Left->1':8s} | "
              f"{'Advantage':9s} | Signal")
        print("-" * 66)
        for level in ["Volume", "AP-set", "AP-ordered", "AP+SP"]:
            pr_r, pr_l, adv = estimate_qt_ps_advantage(
                db, public_corpus, cs_pool, bio_pool, q_vecs,
                n_results=3, level=level, noise_std=noise_std,
                n_repeats=1, trials=200, seed=42,
            )
            if abs(adv) <= 0.01:
                sig = "No separation"
            elif adv >= 0.99:
                sig = "Full separation"
            else:
                sig = "Partial separation"
            print(f"{level:10s} | {pr_r:9.3f} | {pr_l:8.3f} | "
                  f"{adv:+9.3f} | {sig}")


if __name__ == "__main__":
    main()
