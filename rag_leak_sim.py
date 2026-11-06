"""
Encrypted-RAG leakage: starter simulation (CS 666 project)

Pipeline
  1. corpus      -> chunk embeddings X (n x d) with topic labels y  (ground truth, attacker never sees it)
  2. protect     -> DCPE-style transform (SAP or RAP) applied to X and to each query
  3. server      -> top-k search over protected vectors, logs only what leak() allows
  4. leak(L)     -> THE ONLY thing the attacker gets. Enforces the leakage profile.
  5. attack      -> co-retrieval matrix -> spectral clustering -> recovered topic groups
  6. attach      -> a few known chunks name each cluster (stand-in for the public reference corpus)

Run:  python rag_leak_sim.py            (synthetic data, no downloads, ~1 min)
      python rag_leak_sim.py --real     (20 Newsgroups + sentence-transformers, use in Colab)
"""
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- 1. corpus
def synthetic_corpus(n=2000, d=64, n_topics=8, spread=3.0):
    """Gaussian-mixture stand-in for embeddings: one cluster per topic, unit-normalised."""
    centers = rng.normal(size=(n_topics, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    y = rng.integers(0, n_topics, size=n)
    X = centers[y] + spread * rng.normal(size=(n, d)) / np.sqrt(d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, y, centers


def real_corpus(n_topics=8, n=2000):
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    cats = fetch_20newsgroups(subset="train").target_names[:n_topics]
    data = fetch_20newsgroups(subset="train", categories=cats,
                              remove=("headers", "footers", "quotes"))
    keep = [i for i, t in enumerate(data.data) if len(t.split()) > 30][:n]
    texts = [data.data[i] for i in keep]
    y = data.target[keep]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    centers = np.stack([X[y == t].mean(0) for t in range(n_topics)])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return X, y, centers


# ---------------------------------------------------------------- 2. protection
def ball_noise(m, d, radius):
    v = rng.normal(size=(m, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * radius * rng.random((m, 1)) ** (1 / d)


class SAP:
    """Scale-and-perturb: c = s*x + noise.  Secret: scale s.  beta = noise knob."""
    def __init__(self, d, beta=0.1):
        self.s, self.beta = rng.uniform(1.0, 10.0), beta

    def enc(self, X):
        return self.s * X + ball_noise(len(X), X.shape[1], self.s * self.beta)


class RAP:
    """Rotate-and-perturb: c = Q x + noise.  Secret: rotation Q."""
    def __init__(self, d, beta=0.5):
        self.Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        self.beta = beta

    def enc(self, X):
        return X @ self.Q.T + ball_noise(len(X), X.shape[1], self.beta / 4)


# ---------------------------------------------------------------- 3-4. server + leakage
class Server:
    """Holds protected vectors, answers queries, logs. Never sees plaintext or key."""
    def __init__(self, protected, k=10):
        self.C, self.k, self.log = protected, k, []

    def search(self, protected_query):
        d2 = ((self.C - protected_query) ** 2).sum(1)
        ids = np.argpartition(d2, self.k)[: self.k]
        ids = ids[np.argsort(d2[ids])]          # ordered by distance
        self.log.append(ids)
        return ids


def leak(server, level="AP-set"):
    """Enforces the leakage profile L. The attacker code may ONLY call this.
       AP-set     : for each query, the SET of returned ids (order hidden)
       AP-ordered : ids in rank order
    """
    if level == "AP-set":
        return [frozenset(map(int, ids)) for ids in server.log]
    if level == "AP-ordered":
        return [tuple(map(int, ids)) for ids in server.log]
    raise ValueError(level)


# ---------------------------------------------------------------- user (workload)
def make_queries(centers, q, topic_probs, noise=1.5):
    """User asks about topics non-uniformly. Query = noisy point near a topic centre."""
    d = centers.shape[1]
    t = rng.choice(len(centers), size=q, p=topic_probs)
    Z = centers[t] + noise * rng.normal(size=(q, d)) / np.sqrt(d)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


# ---------------------------------------------------------------- 5-6. attack
def attack(logs, n, n_clusters):
    """Co-retrieval structure -> clusters. Uses ONLY the leaked id sets."""
    M = np.zeros((n, n))
    for ids in logs:
        v = np.zeros(n)
        v[list(ids)] = 1
        M += np.outer(v, v)
    np.fill_diagonal(M, 0)
    deg = M.sum(1)
    seen = deg > 0                                  # chunks never returned stay unknown
    # normalised spectral embedding on seen chunks
    Ms = M[np.ix_(seen, seen)]
    dinv = 1 / np.sqrt(Ms.sum(1) + 1e-9)
    L = Ms * dinv[:, None] * dinv[None, :]
    w, V = np.linalg.eigh(L)
    E = V[:, -n_clusters:]
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    lab_seen = KMeans(n_clusters, n_init=10, random_state=0).fit_predict(E)
    labels = -np.ones(n, dtype=int)
    labels[seen] = lab_seen
    return labels, seen


def name_clusters(labels, y, known_per_topic=5):
    """Stand-in for the public reference step: a few known (chunk, topic) pairs name each cluster."""
    mapping = {}
    for c in np.unique(labels[labels >= 0]):
        members = np.where(labels == c)[0]
        pick = rng.choice(members, size=min(known_per_topic, len(members)), replace=False)
        mapping[c] = np.bincount(y[pick]).argmax()
    return mapping


def run(protector="SAP", beta=0.1, qs=(25, 50, 100, 200, 400, 800), real=False, level="AP-set"):
    X, y, centers = real_corpus() if real else synthetic_corpus()
    n, d = X.shape
    n_topics = len(centers)
    probs = rng.dirichlet(np.ones(n_topics) * 2)        # NON-uniform user interest
    prot = (SAP if protector == "SAP" else RAP)(d, beta)
    C = prot.enc(X)
    out = []
    for q in qs:
        srv = Server(C, k=10)
        for z in make_queries(centers, q, probs):
            srv.search(prot.enc(z[None, :])[0])
        labels, seen = attack(leak(srv, level), n, n_topics)
        ari = adjusted_rand_score(y[seen], labels[seen])
        mp = name_clusters(labels, y)
        pred = np.array([mp.get(c, -1) for c in labels])
        acc = (pred[seen] == y[seen]).mean()
        out.append((q, seen.mean(), ari, acc))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    for name, beta in [("SAP", 0.1), ("RAP", 0.5)]:
        print(f"\n{name} beta={beta}  leakage=AP-set   (n=2000 chunks, k=10 returned per query)")
        print(f"{'queries':>8} {'chunks seen':>12} {'cluster ARI':>12} {'topic acc':>10}")
        for q, cov, ari, acc in run(name, beta, real=a.real):
            print(f"{q:>8} {cov:>12.2f} {ari:>12.2f} {acc:>10.2f}")
