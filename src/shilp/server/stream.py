"""Bridge a blocking engine generator to an async one.

`llama.cpp` yields tokens from a C call that blocks the thread it runs on. The
event loop must not be that thread, so the generator is drained in a worker and
its chunks are handed back through an `asyncio.Queue`.

The queue is unbounded: a bounded one would either block the worker (deadlock
risk while the loop is busy) or drop tokens (silent corruption). Backpressure
comes from the HTTP client instead — if it stops reading, TCP stalls and the
socket write in the route blocks.
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

    def pump() -> None:
        try:
            for chunk in source:
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except BaseException as exc:  # forwarded and re-raised in the caller
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    threading.Thread(target=pump, name="shilp-engine", daemon=True).start()

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            return
        if isinstance(item, BaseException):
            raise item
        yield item
