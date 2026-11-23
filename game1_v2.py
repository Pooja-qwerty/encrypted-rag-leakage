"""
Game v1 (Query Indistinguishability on Real Text): Q0 vs Q1 on a fixed CS + BIO DB.
Procedural (no classes) using CS.txt and BIO.txt from the same directory.
"""

from pathlib import Path
import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    HAS_ST = True
except ImportError:
    HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer


def load_chunks_from_files(cs_path="data/CS.txt", bio_path="data/BIO.txt"):
    """Loads chunks split by blank lines from CS.txt and BIO.txt."""
    base_dir = Path(__file__).resolve().parent
    cs_file = Path(cs_path) if Path(cs_path).is_absolute() else base_dir / cs_path
    bio_file = Path(bio_path) if Path(bio_path).is_absolute() else base_dir / bio_path

    texts = []
    subjects = []
    chunk_sizes = []

    for subject, filepath in [("CS", cs_file), ("Bio", bio_file)]:
        raw = filepath.read_text(encoding="utf-8").strip()
        blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
        for block in blocks:
            texts.append(block)
            subjects.append(subject)
            chunk_sizes.append(len(block.encode("utf-8")))

    return {
        "texts": texts,
        "subjects": subjects,
        "chunk_sizes": chunk_sizes,
    }


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


def setup_db(raw_corpus, embed_fn):
    """Builds the fixed search database from CS and Bio chunks."""
    vectors = embed_fn(raw_corpus["texts"])
    return {
        "texts": list(raw_corpus["texts"]),
        "subjects": list(raw_corpus["subjects"]),
        "chunk_sizes": list(raw_corpus["chunk_sizes"]),
        "vectors": vectors,
    }


def top_n(db, query_vec, n_results=3):
    """Returns indices of top-N most similar chunks in descending score order."""
    q = np.asarray(query_vec, dtype=float)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    scores = db["vectors"] @ q
    k = min(n_results, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [int(i) for i in idx]


def query_key(query_text):
    """Exact canonical query key for repeat-token tracking in AP+SP."""
    return query_text.encode("utf-8")


def leak(db, seen, query_text, query_vec, n_results=3, level="AP-set"):
    """Enforces leakage profile L: outputs only what L allows."""
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

    raise ValueError(f"Unknown leakage level: {level}")


def initialize_game1(db, side):
    """Initializes fixed-DB query game: Left uses Q0, Right uses Q1."""
    state = {
        "side": side,
        "db": db,
        "log": [],
        "seen": {},
    }
    l_setup = {"n": len(db["texts"]), "chunk_sizes": list(db["chunk_sizes"])}
    return state, l_setup


def search_game1(state, query_text, query_vec, n_results=3, level="AP-set"):
    """Evaluates query in the active world and appends leakage to log."""
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
    """Returns adversary guess bit d in {0, 1}."""
    return d


def adversary_guess_game1(log, q0_signature_ids, q1_signature_ids, level="AP-set"):
    """
    Infers whether the observed query stream matches Q1 (1) or Q0 (0)
    from leaked logs and the adversary's known reference chunk sets.
    """
    if not log or "chunk_ids" not in log[0]:
        return 0

    q0_set = set(q0_signature_ids)
    q1_set = set(q1_signature_ids)

    token_counts = {}
    if level == "AP+SP":
        for entry in log:
            tok = entry.get("repeat_token")
            if tok is not None:
                token_counts[tok] = token_counts.get(tok, 0) + 1

    score_q1 = 0.0
    score_q0 = 0.0

    for entry in log:
        obs = entry["chunk_ids"]
        weight = 1.0
        if level == "AP+SP":
            tok = entry.get("repeat_token")
            if tok is not None and token_counts.get(tok, 1) > 1:
                weight = 1.25

        if isinstance(obs, list) and level in ("AP-ordered", "AP+SP"):
            for rank, cid in enumerate(obs):
                rank_w = weight / (1.0 + rank)
                if cid in q1_set:
                    score_q1 += rank_w
                if cid in q0_set:
                    score_q0 += rank_w
        else:
            for cid in obs:
                if cid in q1_set:
                    score_q1 += weight
                if cid in q0_set:
                    score_q0 += weight

    return 1 if score_q1 > score_q0 else 0


def build_query_stream(q_text, repeat_indices=(3, 6), n_steps=8):
    """Builds a sequence of queries with exact repeats at repeat_indices."""
    repeat_set = set(repeat_indices)
    stream = []
    for step in range(n_steps):
        if step in repeat_set:
            stream.append(q_text)
        else:
            stream.append(q_text)
    return stream


def play_query_game(side, db, q0_text, q1_text, q_vecs, q0_sig, q1_sig, n_results=3, level="AP-set"):
    """Runs one Left (Q0) or Right (Q1) game on the fixed DB."""
    state, _l_setup = initialize_game1(db, side=side)
    chosen_q = q0_text if side == "left" else q1_text
    chosen_vec = q_vecs[chosen_q]

    stream = build_query_stream(chosen_q, repeat_indices=(2, 5), n_steps=8)
    for q_txt in stream:
        search_game1(state, q_txt, chosen_vec, n_results=n_results, level=level)

    d = adversary_guess_game1(state["log"], q0_sig, q1_sig, level=level)
    return finalize(d)


def estimate_query_advantage(db, q0_text, q1_text, q_vecs, n_results=3, level="AP-set", trials=100):
    """Computes Adv = Pr[A(Right/Q1) -> 1] - Pr[A(Left/Q0) -> 1] on the fixed DB."""
    q0_sig = top_n(db, q_vecs[q0_text], n_results=n_results)
    q1_sig = top_n(db, q_vecs[q1_text], n_results=n_results)

    right_to_1 = 0
    left_to_1 = 0

    for _ in range(trials):
        if play_query_game("left", db, q0_text, q1_text, q_vecs, q0_sig, q1_sig, n_results, level) == 1:
            left_to_1 += 1
        if play_query_game("right", db, q0_text, q1_text, q_vecs, q0_sig, q1_sig, n_results, level) == 1:
            right_to_1 += 1

    pr_r = right_to_1 / trials
    pr_l = left_to_1 / trials
    return pr_r, pr_l, pr_r - pr_l


def main():
    raw_corpus = load_chunks_from_files("data/CS.txt", "data/BIO.txt")
    embed_fn = build_embedder(raw_corpus["texts"])
    db = setup_db(raw_corpus, embed_fn)

    # Both queries touch "indexing" so they aren't trivially separated by a single word
    q0_text = "Explain indexing in a database."
    q1_text = "Explain gene indexing."

    q_vecs_arr = embed_fn([q0_text, q1_text])
    q_vecs = {
        q0_text: q_vecs_arr[0],
        q1_text: q_vecs_arr[1],
    }

    print(f"Loaded {len(db['texts'])} chunks ({db['subjects'].count('CS')} CS, {db['subjects'].count('Bio')} Bio)")
    print(f"Q0 (Left):  {q0_text!r} -> top {3} chunks: {top_n(db, q_vecs[q0_text], n_results=3)}")
    print(f"Q1 (Right): {q1_text!r} -> top {3} chunks: {top_n(db, q_vecs[q1_text], n_results=3)}\n")

    print(f"{'Leakage':10s} | {'Guess Q1 on Right':17s} | {'Guess Q1 on Left':16s} | {'Advantage':9s} | {'Signal'}")
    print("-" * 78)

    for level in ["Volume", "AP-set", "AP-ordered", "AP+SP"]:
        pr_r, pr_l, adv = estimate_query_advantage(
            db, q0_text, q1_text, q_vecs, n_results=3, level=level, trials=100
        )
        if abs(adv) >= 0.99:
            signal = "Full separation (Q1 vs Q0)"
        elif abs(adv) <= 0.01:
            signal = "No separation (indistinguishable)"
        else:
            signal = "Partial separation"

        print(f"{level:10s} | {pr_r:17.3f} | {pr_l:16.3f} | {adv:+9.3f} | {signal}")


if __name__ == "__main__":
    main()