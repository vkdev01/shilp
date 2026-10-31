"""Prometheus text metrics, without the client library.

Four counters and two summaries is not worth a dependency, and hand-writing the
exposition format makes clear what is actually being counted:

    shilp_requests_total{surface,endpoint,model}       counter
    shilp_tokens_total{model,direction}                counter
    shilp_request_duration_ms{model}                   sum + count
    shilp_time_to_first_token_ms{model}                sum + count
    shilp_models_resident                              gauge (set by the loader)
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Iterable

Labels = tuple[tuple[str, str], ...]


def _labels(pairs: dict[str, str]) -> str:
    if not pairs:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in sorted(pairs.items()))
    return "{" + inner + "}"


def _key(pairs: dict[str, str]) -> Labels:
    return tuple(sorted(pairs.items()))


class Metrics:
    """Counters and running sums, all guarded by one lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, Labels], float] = defaultdict(float)
        self._sums: dict[tuple[str, Labels], list[float]] = defaultdict(lambda: [0.0, 0.0])
        self._gauges: dict[str, float] = {}

    # --- recording -----------------------------------------------------
    def inc(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        with self._lock:
            self._counters[(name, _key(labels or {}))] += value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            bucket = self._sums[(name, _key(labels or {}))]
            bucket[0] += value
            bucket[1] += 1

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    # --- exposition ----------------------------------------------------
    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{_labels(dict(labels))} {value:g}")
            for (name, labels), (total, count) in sorted(self._sums.items()):
                label_dict = dict(labels)
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_sum{_labels(label_dict)} {total:g}")
                lines.append(f"{name}_count{_labels(label_dict)} {count:g}")
            for name, value in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value:g}")
        return "\n".join(lines) + "\n"

    def snapshot(self) -> Iterable[str]:
        return self.render().splitlines()
