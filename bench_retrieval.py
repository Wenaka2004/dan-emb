"""Benchmark: numpy vs faiss-cpu (flat + IVF) for 137k×4096 retrieval."""

import time
import numpy as np
import faiss

# Load embeddings
print("Loading embeddings...")
embeddings = np.load("embedding_index/embeddings.npy").astype(np.float32)
n, d = embeddings.shape
print(f"  Shape: {n}×{d}, dtype: {embeddings.dtype}")

# Normalize (should already be, but ensure)
faiss.normalize_L2(embeddings)

# Generate random queries (simulate 100 user queries)
np.random.seed(42)
n_queries = 100
queries = np.random.randn(n_queries, d).astype(np.float32)
faiss.normalize_L2(queries)

K = 20  # top-k

print(f"\nBenchmark: {n_queries} queries, top-{K}, {n} vectors × {d} dims\n")

# ── 1. Numpy brute force ───────────────────────────────────────────────────
print("1. Numpy (dot product = cosine for normalized vectors)")
# Warmup
_ = queries[:1] @ embeddings.T

t0 = time.perf_counter()
scores = queries @ embeddings.T
indices = np.argpartition(-scores, K, axis=1)[:, :K]
# Sort the top-k
for i in range(n_queries):
    idx = indices[i]
    order = np.argsort(-scores[i, idx])
    indices[i] = idx[order]
t_numpy = time.perf_counter() - t0
print(f"   Total: {t_numpy*1000:.1f}ms, Per query: {t_numpy/n_queries*1000:.2f}ms")

# ── 2. Faiss IndexFlatIP (exact search, optimized) ─────────────────────────
print("\n2. Faiss IndexFlatIP (exact, CPU-optimized)")
index_flat = faiss.IndexFlatIP(d)
index_flat.add(embeddings)

# Warmup
_ = index_flat.search(queries[:1], K)

t0 = time.perf_counter()
D_flat, I_flat = index_flat.search(queries, K)
t_flat = time.perf_counter() - t0
print(f"   Total: {t_flat*1000:.1f}ms, Per query: {t_flat/n_queries*1000:.2f}ms")
print(f"   vs numpy: {t_numpy/t_flat:.1f}x faster")

# Verify results match
print(f"   Results match numpy: {np.array_equal(I_flat, indices)}")

# ── 3. Faiss IVFFlat (approximate, faster for large datasets) ──────────────
print("\n3. Faiss IVFFlat (approximate, nlist=256, nprobe=32)")
nlist = 256
quantizer = faiss.IndexFlatIP(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

# Train
t0 = time.perf_counter()
index_ivf.train(embeddings)
t_train = time.perf_counter() - t0
print(f"   Train time: {t_train:.2f}s")

index_ivf.add(embeddings)
index_ivf.nprobe = 32

# Warmup
_ = index_ivf.search(queries[:1], K)

t0 = time.perf_counter()
D_ivf, I_ivf = index_ivf.search(queries, K)
t_ivf = time.perf_counter() - t0
print(f"   Total: {t_ivf*1000:.1f}ms, Per query: {t_ivf/n_queries*1000:.2f}ms")
print(f"   vs numpy: {t_numpy/t_ivf:.1f}x faster")

# Recall check (vs exact flat)
recall = 0
for i in range(n_queries):
    recall += len(set(I_flat[i]) & set(I_ivf[i])) / K
recall /= n_queries
print(f"   Recall@{K}: {recall:.4f}")

# ── 4. Faiss IVFPQ (compressed, even faster, lower memory) ────────────────
print("\n4. Faiss IVFPQ (approximate + compressed, nlist=256, m=64, nprobe=32)")
m = 64  # number of subquantizers (must divide d=4096)
quantizer2 = faiss.IndexFlatIP(d)
index_pq = faiss.IndexIVFPQ(quantizer2, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)  # 8 bits per sub

t0 = time.perf_counter()
index_pq.train(embeddings)
t_train_pq = time.perf_counter() - t0
print(f"   Train time: {t_train_pq:.2f}s")

index_pq.add(embeddings)
index_pq.nprobe = 32

# Warmup
_ = index_pq.search(queries[:1], K)

t0 = time.perf_counter()
D_pq, I_pq = index_pq.search(queries, K)
t_pq = time.perf_counter() - t0
print(f"   Total: {t_pq*1000:.1f}ms, Per query: {t_pq/n_queries*1000:.2f}ms")
print(f"   vs numpy: {t_numpy/t_pq:.1f}x faster")

recall_pq = 0
for i in range(n_queries):
    recall_pq += len(set(I_flat[i]) & set(I_pq[i])) / K
recall_pq /= n_queries
print(f"   Recall@{K}: {recall_pq:.4f}")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'Method':<25} {'Total (ms)':<12} {'Per query (ms)':<16} {'vs numpy':<10} {'Recall@'+str(K)}")
print(f"{'='*60}")
print(f"{'Numpy brute force':<25} {t_numpy*1000:<12.1f} {t_numpy/n_queries*1000:<16.2f} {'1.0x':<10} {'1.0000'}")
print(f"{'Faiss IndexFlatIP':<25} {t_flat*1000:<12.1f} {t_flat/n_queries*1000:<16.2f} {t_numpy/t_flat:<10.1f} {'1.0000'}")
print(f"{'Faiss IVFFlat':<25} {t_ivf*1000:<12.1f} {t_ivf/n_queries*1000:<16.2f} {t_numpy/t_ivf:<10.1f} {recall:<10.4f}")
print(f"{'Faiss IVFPQ':<25} {t_pq*1000:<12.1f} {t_pq/n_queries*1000:<16.2f} {t_numpy/t_pq:<10.1f} {recall_pq:<10.4f}")
