"""Build embedding index from Danbooru wiki data with concurrent rate-limited API calls."""

import json
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, EMBEDDING_MODEL, INDEX_DIR
from dtext import build_embedding_text

# ── Config ──────────────────────────────────────────────────────────────────

API_URL = f"{SILICONFLOW_BASE_URL}/embeddings"
MODEL = EMBEDDING_MODEL

PARQUET_PATH = "danbooru_wiki.parquet"
OUTPUT_DIR = Path(INDEX_DIR)

BATCH_SIZE = 200
MAX_CONCURRENT = 20
TARGET_RATIO = 0.85
MAX_RPM = 2000
MAX_TPM = 1_000_000
CHARS_PER_TOKEN = 3.5

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2

# ── Thread-safe Rate Limiter ────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, max_rpm: int, max_tpm: int, target_ratio: float = 0.85):
        self.target_rpm = int(max_rpm * target_ratio)
        self.target_tpm = int(max_tpm * target_ratio)
        self._lock = threading.Lock()
        self._request_times: deque = deque()
        self._token_events: deque = deque()

    def _prune(self, now: float, window: float = 60.0):
        cutoff = now - window
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()
        while self._token_events and self._token_events[0][0] < cutoff:
            self._token_events.popleft()

    def get_stats(self) -> tuple[int, int]:
        with self._lock:
            self._prune(time.time())
            return len(self._request_times), sum(t for _, t in self._token_events)

    def wait_and_acquire(self, estimated_tokens: int):
        while True:
            with self._lock:
                now = time.time()
                self._prune(now)
                rpm = len(self._request_times)
                tpm = sum(t for _, t in self._token_events)
                if rpm < self.target_rpm and tpm + estimated_tokens <= self.target_tpm:
                    self._request_times.append(now)
                    self._token_events.append((now, estimated_tokens))
                    return
                wait = 0.5
                if rpm >= self.target_rpm:
                    wait = max(wait, self._request_times[0] + 60.1 - now)
                if tpm + estimated_tokens > self.target_tpm:
                    wait = max(wait, self._token_events[0][0] + 60.1 - now)
            time.sleep(min(wait, 2))

    def update_tokens(self, actual_tokens: int, estimated_tokens: int):
        with self._lock:
            for i in range(len(self._token_events) - 1, -1, -1):
                if self._token_events[i][1] == estimated_tokens:
                    ts = self._token_events[i][0]
                    self._token_events[i] = (ts, actual_tokens)
                    return

# ── API Client ──────────────────────────────────────────────────────────────

def embed_batch(texts: list[str], batch_idx: int) -> tuple[int, list[list[float]], int]:
    payload = {"model": MODEL, "input": texts, "encoding_format": "float"}
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return batch_idx, [item["embedding"] for item in data["data"]], data["usage"]["total_tokens"]
        except requests.exceptions.HTTPError:
            status = resp.status_code
            if status == 429 or status >= 500:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  [batch {batch_idx}] {status}, retry {attempt+1} in {delay}s", flush=True)
                time.sleep(delay)
            else:
                print(f"  [batch {batch_idx}] {status}: {resp.text[:200]}", flush=True)
                raise
        except requests.exceptions.RequestException as e:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"  [batch {batch_idx}] network error, retry {attempt+1} in {delay}s", flush=True)
            time.sleep(delay)

    raise RuntimeError(f"Batch {batch_idx} failed after {MAX_RETRIES} retries")


# ── Data Loading ────────────────────────────────────────────────────────────

def load_and_prepare(parquet_path: str) -> pd.DataFrame:
    df = pq.read_table(parquet_path).to_pandas()
    print(f"Loaded {len(df)} entries from {parquet_path}")

    mask = df["body"].notna() & (df["body"].str.strip() != "")
    df = df[mask].copy()
    print(f"After filtering empty body: {len(df)} entries")

    df = df[~df["is_deleted"]].copy()
    print(f"After filtering deleted: {len(df)} entries")

    df["embed_text"] = df.apply(build_embedding_text, axis=1)

    min_useful_len = len("Tag: x\nCategory: x") + 10
    df = df[df["embed_text"].str.len() >= min_useful_len].copy()
    print(f"After filtering too-short entries: {len(df)} entries")

    df = df.reset_index(drop=True)
    return df

# ── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    df = load_and_prepare(PARQUET_PATH)
    total = len(df)

    batches = []
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        texts = df.loc[start:end - 1, "embed_text"].tolist()
        est_tokens = sum(len(t) for t in texts) // CHARS_PER_TOKEN
        batches.append((start, end, texts, est_tokens))

    embeddings_path = OUTPUT_DIR / "embeddings.npy"
    completed_path = OUTPUT_DIR / "completed.json"

    start_batch = 0
    existing = None
    if embeddings_path.exists() and completed_path.exists():
        existing = np.load(embeddings_path)
        with open(completed_path) as f:
            completed_batches = json.load(f).get("completed_batches", 0)
        start_batch = completed_batches
        print(f"Resuming: {completed_batches} batches ({len(existing)} entries) already done")

    if start_batch >= len(batches):
        print("All batches already embedded!")
        return

    remaining = batches[start_batch:]
    print(f"\n{len(batches)} total batches, {len(remaining)} remaining")
    print(f"Rate targets: {int(MAX_RPM * TARGET_RATIO)} RPM, {int(MAX_TPM * TARGET_RATIO)} TPM")
    print(f"Batch size: {BATCH_SIZE}, Concurrent: {MAX_CONCURRENT}\n")

    limiter = RateLimiter(MAX_RPM, MAX_TPM, TARGET_RATIO)
    results = {}
    results_lock = threading.Lock()
    completed_count = start_batch

    t_start = time.time()
    last_print = 0

    def submit_with_rate_limit(executor, batch_idx, start, end, texts, est_tokens):
        limiter.wait_and_acquire(est_tokens)
        return executor.submit(embed_batch, texts, batch_idx), est_tokens

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        batch_iter = iter(enumerate(remaining, start=start_batch))

        futures = {}
        for batch_idx, (start, end, texts, est_tokens) in batch_iter:
            future, est_tok = submit_with_rate_limit(executor, batch_idx, start, end, texts, est_tokens)
            futures[future] = (batch_idx, start, end, est_tok)
            if len(futures) >= MAX_CONCURRENT:
                break

        while futures:
            done = next(as_completed(futures))
            batch_idx, start, end, est_tok = futures.pop(done)

            try:
                _, embeddings, actual_tokens = done.result()
            except Exception as e:
                print(f"  FATAL: batch {batch_idx} failed: {e}", flush=True)
                raise

            limiter.update_tokens(actual_tokens, est_tok)

            with results_lock:
                results[batch_idx] = (start, embeddings)
                completed_count += 1

            for next_idx, (ns, ne, nt, nest) in batch_iter:
                future, est = submit_with_rate_limit(executor, next_idx, ns, ne, nt, nest)
                futures[future] = (next_idx, ns, ne, est)
                break

            now = time.time()
            if now - last_print >= 1.0:
                done_entries = min(completed_count * BATCH_SIZE, total)
                speed = done_entries / (now - t_start) * 60 if now - t_start > 0 else 0
                eta = (total - done_entries) / speed * 60 if speed > 0 else 0
                rpm, tpm = limiter.get_stats()
                print(
                    f"  [{time.strftime('%H:%M:%S')}] "
                    f"{done_entries}/{total} ({done_entries/total*100:.1f}%) | "
                    f"RPM: {rpm}/{limiter.target_rpm} | "
                    f"TPM: {tpm/1000:.0f}K/{limiter.target_tpm/1000:.0f}K | "
                    f"Speed: {speed:.0f} entries/min | "
                    f"ETA: {eta/60:.1f}min",
                    flush=True,
                )
                last_print = now

            if completed_count % 50 == 0:
                _save_progress(results, existing, embeddings_path, completed_path, completed_count, df)

    _save_progress(results, existing, embeddings_path, completed_path, completed_count, df, final=True)
    print(f"\nDone in {(time.time() - t_start)/60:.1f} min")


def _save_progress(results, existing, embeddings_path, completed_path, completed_batches, df, final=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_arrays = []
    if existing is not None:
        all_arrays.append(existing)
    for idx in sorted(results.keys()):
        _, embeddings = results[idx]
        all_arrays.append(np.array(embeddings))

    if all_arrays:
        combined = np.vstack(all_arrays)
        np.save(embeddings_path, combined)

        if final:
            meta = df[["tag", "title", "category", "other_names", "body"]].copy()
            meta["body_clean"] = df["embed_text"]
            meta = meta.iloc[:len(combined)]
            meta.to_parquet(OUTPUT_DIR / "metadata.parquet", index=False)
            print(f"Saved: embeddings {combined.shape}, metadata {len(meta)} entries")
            if completed_path.exists():
                completed_path.unlink()
        else:
            print(f"  Checkpoint: {len(combined)} entries saved", flush=True)

    with open(completed_path, "w") as f:
        json.dump({"completed_batches": completed_batches}, f)


if __name__ == "__main__":
    main()
