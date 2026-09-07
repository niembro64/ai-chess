"""Source-reserved JESTER replay; long games cannot evict short curriculum.

Quotas describe optimizer samples, not the number of games started. Unknown
caps get at most 5% of a batch and never supply value labels. Missing known
sources share their quota proportionally until those sources become available.
"""

import random

import numpy as np

from .model import NUM_PLANES
from .selfplay import ReplayBuffer


class JesterReplayBuffer:
    weights = {"curriculum": .30, "bridge": .30, "competitive": .25, "bootstrap": .10, "cap": .05}

    def __init__(self, capacity, num_planes=NUM_PLANES):
        self.capacity = capacity
        self.buckets = {
            key: ReplayBuffer(max(1, int(capacity * weight)), num_planes)
            for key, weight in self.weights.items()
        }
        self.total_added = 0
        self.last_sample_counts = {}

    def __len__(self):
        return sum(map(len, self.buckets.values()))

    @property
    def ready(self):
        return any(len(b) for key, b in self.buckets.items() if key != "cap")

    @property
    def source_sizes(self):
        return {key: len(b) for key, b in self.buckets.items()}

    def add(self, example):
        key = "cap" if not example.outcome_known else example.source
        self.buckets[key if key in self.buckets else "competitive"].add(example)
        self.total_added += 1

    def sample(self, batch_size, rng=None):
        rng = rng or random
        available = [key for key, b in self.buckets.items() if len(b) and key != "cap"]
        if not available:
            raise ValueError("JESTER replay needs an observed outcome before training")
        counts = {key: 0 for key in self.buckets}
        counts["cap"] = int(batch_size * self.weights["cap"]) if len(self.buckets["cap"]) else 0
        remaining = batch_size - counts["cap"]
        total_weight = sum(self.weights[key] for key in available)
        exact = {key: remaining * self.weights[key] / total_weight for key in available}
        for key in available:
            counts[key] = int(exact[key])
        # Largest-remainder allocation keeps quotas reproducible, including small batches.
        for key in sorted(available, key=lambda k: exact[k] - counts[k], reverse=True)[:batch_size - sum(counts.values())]:
            counts[key] += 1
        parts = [self.buckets[key].sample(n, rng) for key, n in counts.items() if n]
        order = list(range(batch_size))
        rng.shuffle(order)
        self.last_sample_counts = counts
        return tuple(np.concatenate([p[i] for p in parts])[order] for i in range(5))
