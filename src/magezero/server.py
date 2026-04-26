import os
import threading
import time
from collections import defaultdict
from queue import Queue, Empty

import numpy as np
import torch
import waitress
from pyroaring import BitMap
from flask import Flask, request, Response
import msgpack
from model import Net, load_model, GLOBAL_MAX, ACTIONS_MAX

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Threading config
TORCH_THREADS = 1 #max(1, os.cpu_count() // 2)
torch.set_num_threads(TORCH_THREADS)

# Batching config.
# MAX_WAIT_MS = 0 disables real coalescing — the worker drains whatever is
# already in the queue at the instant the first request arrives, which on a
# fast localhost loop almost always means batch_size=1. A small (~5ms) wait
# lets concurrent MCTS requests (up to MAX_PENDING per tree on the JVM side)
# accumulate into one forward pass. Sparse EmbeddingBag at batch=1 is memory-
# bandwidth bound on CPU, so this is typically a 5-15x throughput win.
# See madsbolaris/MageZero#1 (P1) and madsbolaris/mage#1 (J1 — must pair).
MAX_BATCH = 32
MAX_WAIT_MS = 5

# Logging control — set MZ_DEBUG_INFERENCE=1 to see per-request/batch prints.
DEBUG_INFERENCE = os.getenv("MZ_DEBUG_INFERENCE") == "1"

#module state
server_model = None
IGNORE_BM = None
VALID_RANGE = None

app = Flask(__name__)

req_counter = 0
req_counter_lock = threading.Lock()

# ─── metrics ─────────────────────────────────────────────────

class ServerMetrics:
    """Thread-safe throughput and latency counters for the inference server."""
    __slots__ = ("_lock", "total_requests", "total_batches", "total_inferences",
                 "batch_size_counts", "latency_samples", "_start_time")

    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests = 0
        self.total_batches = 0
        self.total_inferences = 0  # total bag rows (>= requests when multi-bag)
        self.batch_size_counts: dict[int, int] = defaultdict(int)
        self.latency_samples: list[float] = []  # ms, capped at 10k
        self._start_time = time.perf_counter()

    def record_batch(self, batch_size: int, total_bags: int):
        with self._lock:
            self.total_batches += 1
            self.total_requests += batch_size
            self.total_inferences += total_bags
            self.batch_size_counts[batch_size] += 1

    def record_latency(self, ms: float):
        with self._lock:
            if len(self.latency_samples) < 100_000:
                self.latency_samples.append(ms)

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = time.perf_counter() - self._start_time
            samples = sorted(self.latency_samples)
            n = len(samples)
            return {
                "elapsed_sec": round(elapsed, 2),
                "total_requests": self.total_requests,
                "total_batches": self.total_batches,
                "total_inferences": self.total_inferences,
                "requests_per_sec": round(self.total_requests / elapsed, 2) if elapsed > 0 else 0,
                "inferences_per_sec": round(self.total_inferences / elapsed, 2) if elapsed > 0 else 0,
                "batches_per_sec": round(self.total_batches / elapsed, 2) if elapsed > 0 else 0,
                "batch_size_distribution": dict(sorted(self.batch_size_counts.items())),
                "latency_p50_ms": round(samples[n // 2], 2) if n else 0,
                "latency_p95_ms": round(samples[int(n * 0.95)], 2) if n else 0,
                "latency_p99_ms": round(samples[int(n * 0.99)], 2) if n else 0,
                "latency_samples": n,
            }

    def reset(self):
        with self._lock:
            self.total_requests = 0
            self.total_batches = 0
            self.total_inferences = 0
            self.batch_size_counts.clear()
            self.latency_samples.clear()
            self._start_time = time.perf_counter()


metrics = ServerMetrics()

def init(deck: str, version: int, port: int):
    global server_model, IGNORE_BM, VALID_RANGE

    model_dir = f"models/{deck}/ver{version}"
    ignore_path = f"{model_dir}/ignore.roar"
    model_path = f"{model_dir}/model.pt.gz"

    with open(ignore_path, "rb") as f:
        IGNORE_BM = BitMap.deserialize(f.read())

    VALID_RANGE = BitMap(range(GLOBAL_MAX))

    server_model = Net(GLOBAL_MAX, ACTIONS_MAX).to(DEVICE).eval()
    ckpt = load_model(model_path)
    server_model.load_state_dict(ckpt["model_state_dict"])

    threading.Thread(target=worker_loop, daemon=True).start()

    print(f"[INIT] deck={deck} ver={version} port={port} device={DEVICE}")
    waitress.serve(app, host="127.0.0.1", port=port, threads=6)

class Pending:
    __slots__ = ("idx", "off", "evt", "out", "req_id", "pre_count", "post_count", "t_recv", "t_done", "num_bags")

    def __init__(self, req_id, indices, offsets):
        self.req_id = req_id
        self.pre_count = len(indices)
        self.t_recv = time.perf_counter()
        self.evt = threading.Event()
        self.out = None
        self.t_done = 0.0

        indices, offsets, num_bags = apply_ignore(indices, offsets)
        self.post_count = len(indices)
        self.num_bags = num_bags

        self.idx = torch.tensor(indices, dtype=torch.long)
        self.off = torch.tensor(offsets, dtype=torch.long)


def apply_ignore(indices: list[int], offsets: list[int] | None):
    if not offsets:
        offsets = [0]

    if len(offsets) == 1:
        # Single bag - pure bitmap ops in C
        kept_bm = (BitMap(indices) - IGNORE_BM) & VALID_RANGE
        return list(kept_bm), [0], 1

    # Multi-bag
    n = len(indices)
    new_indices = []
    new_offsets = [0]

    for b in range(len(offsets)):
        start = offsets[b]
        end = offsets[b + 1] if b + 1 < len(offsets) else n

        kept_bm = (BitMap(indices[start:end]) - IGNORE_BM) & VALID_RANGE
        new_indices.extend(kept_bm)
        new_offsets.append(len(new_indices))

    new_offsets = new_offsets[:-1]
    return new_indices, new_offsets, len(new_offsets)


Q: "Queue[Pending]" = Queue(maxsize=4096)


def worker_loop():
    while True:
        p0 = Q.get()
        batch = [p0]

        # Collect more requests up to MAX_BATCH within MAX_WAIT_MS.
        # Fixed: use `and` not `or` to respect batch ceiling (C5).
        deadline = time.perf_counter() + (MAX_WAIT_MS / 1000.0)
        while len(batch) < MAX_BATCH:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(Q.get(timeout=remaining))
            except Empty:
                break

        bag_counts = [p.num_bags for p in batch]

        # Fast path: single request
        if len(batch) == 1:
            idx = batch[0].idx.to(DEVICE, non_blocking=True)
            off = batch[0].off.to(DEVICE, non_blocking=True)
        else:
            # Concatenate indices
            idx = torch.cat([p.idx for p in batch]).to(DEVICE, non_blocking=True)

            # Vectorized offset adjustment
            all_off = torch.cat([p.off for p in batch])
            idx_lens = torch.tensor([len(p.idx) for p in batch])
            bag_counts_t = torch.tensor(bag_counts)
            adjustments = torch.repeat_interleave(
                torch.cat([torch.tensor([0]), idx_lens.cumsum(0)[:-1]]),
                bag_counts_t
            )
            off = (all_off + adjustments).to(DEVICE, non_blocking=True)

        # Single forward pass
        with torch.no_grad():
            pA, pB, tgt, bin2, val = server_model(idx, off)

        # Move to CPU once
        pA = pA.cpu()
        pB = pB.cpu()
        tgt = tgt.cpu()
        bin2 = bin2.cpu()
        val = val.cpu()

        # Split results back to individual requests.
        # Policy/value tensors are sent as raw float32 binary blobs instead of
        # Python float lists — avoids .tolist() Python-object overhead and
        # msgpack float64 encoding. Java client reads little-endian float32.
        # See madsbolaris/MageZero#2 (P2).
        row = 0
        for p, num_bags in zip(batch, bag_counts):
            if num_bags == 1:
                p.out = {
                    "policy_player": pA[row].numpy().tobytes(),
                    "policy_opponent": pB[row].numpy().tobytes(),
                    "policy_target": tgt[row].numpy().tobytes(),
                    "policy_binary": bin2[row].numpy().tobytes(),
                    "value": float(val[row].item()),
                }
            else:
                p.out = [
                    {
                        "policy_player": pA[row + i].numpy().tobytes(),
                        "policy_opponent": pB[row + i].numpy().tobytes(),
                        "policy_target": tgt[row + i].numpy().tobytes(),
                        "policy_binary": bin2[row + i].numpy().tobytes(),
                        "value": float(val[row + i].item()),
                    }
                    for i in range(num_bags)
                ]
            row += num_bags
            p.t_done = time.perf_counter()
            p.evt.set()

        metrics.record_batch(len(batch), row)
        if DEBUG_INFERENCE:
            print(f"[BATCH] size={len(batch)}, total_bag_size={row}")


@app.post("/evaluate")
def evaluate():
    global req_counter

    data = msgpack.unpackb(request.data, raw=False)
    with req_counter_lock:
        req_counter += 1

    indices = data.get("indices", [])
    offsets = data.get("offsets", [])
    pending = Pending(req_counter, indices, offsets)

    if DEBUG_INFERENCE:
        print(f"[REQ {pending.req_id}] indices={pending.pre_count}, kept={pending.post_count}, bag_size={pending.num_bags}")

    Q.put(pending)
    pending.evt.wait()

    total_ms = (pending.t_done - pending.t_recv) * 1000.0
    metrics.record_latency(total_ms)
    if DEBUG_INFERENCE:
        print(f"[REQ {pending.req_id}] done: {total_ms:.1f}ms")

    return Response(msgpack.packb(pending.out, use_bin_type=True), mimetype="application/x-msgpack")


@app.get("/healthz")
def healthz():
    return "ok", 200


@app.get("/metrics")
def get_metrics():
    """Return server throughput and latency metrics as JSON."""
    return metrics.snapshot()


@app.post("/metrics/reset")
def reset_metrics():
    """Reset all metrics counters (useful for benchmark isolation)."""
    metrics.reset()
    return {"status": "reset"}


if __name__ == "__main__":
    import argparse
    import waitress
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--port", type=int, default=50052)
    args = parser.parse_args()
    init(args.deck, args.version, args.port)