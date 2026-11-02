"""Bridge a blocking engine generator to an async one.

`llama.cpp` yields tokens from a C call that blocks the thread it runs on. The
event loop must not be that thread, so the generator is drained in a worker and
its chunks are handed back through an `asyncio.Queue`.

The queue is unbounded: a bounded one would either block the worker (deadlock
risk while the loop is busy) or drop tokens (silent corruption). Backpressure
comes from the HTTP client instead — if it stops reading, TCP stalls and the
socket write in the route blocks.

Cancellation runs the other way. When the consumer goes away — the client
disconnected, so Starlette cancelled the response — a flag tells the worker to
stop at the next chunk and close the source generator *from its own thread*.
Closing it runs the engine's `finally`, which stops llama.cpp and releases the
engine lock. Without this, an abandoned answer kept generating to the end and
every request queued behind it waited for tokens nobody would read.
"""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator, Iterator

from ..engine.base import Chunk

_SENTINEL = object()


async def to_async(source: Iterator[Chunk]) -> AsyncIterator[Chunk]:
    """Yield the chunks of a blocking iterator without blocking the loop."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()

    def deliver(item: object) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            pass  # the loop is gone; nobody is listening

    def pump() -> None:
        try:
            for chunk in source:
                if stop.is_set():
                    break
                deliver(chunk)
        except BaseException as exc:  # forwarded and re-raised in the caller
            deliver(exc)
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()
            deliver(_SENTINEL)

    worker = threading.Thread(target=pump, name="shilp-engine", daemon=True)
    worker.start()

    try:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # Normal completion, an error, or cancellation: in every case the worker
        # must not keep generating on our behalf.
        stop.set()
