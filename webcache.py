#!/usr/bin/env python3
"""Short-lived memory of what the web already told this process.

A follow-up question, a regenerated answer and a second search round all
tend to land on the pages the last turn already read. Each of those used to
fetch the page again, extract it again and pay the distiller again — seconds
per page on a small box, and one more request to a search engine that rate
limits anything that looks automated.

Three caches, all in memory and all bounded:

* **pages** — a fetched page after extraction and before it is cut down to a
  question, keyed by its normalised URL. Cut-down copies differ per question,
  so the cut happens after the cache, never before.
* **searches** — a query's results, for a shorter time than pages: results
  move faster than the pages behind them.
* **distilled** — what the distiller copied out of a page for a question,
  keyed by the page's exact text so a changed page is never answered from an
  old reading.

Nothing here survives a restart, on purpose: a cache that outlives the
process is one that has to be invalidated, and this one only has to be fast.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict
from typing import Any, Hashable, Optional

from config import get_web_cache_minutes


class TTLCache:
    """A small LRU with an expiry, safe across the request threads."""

    def __init__(self, max_entries: int, ttl_scale: float = 1.0) -> None:
        self._max = max_entries
        self._scale = ttl_scale
        self._lock = threading.Lock()
        self._items: "OrderedDict[Hashable, tuple]" = OrderedDict()

    def _ttl(self) -> float:
        return get_web_cache_minutes() * 60.0 * self._scale

    def get(self, key: Hashable) -> Optional[Any]:
        """A copy of what was stored, or None if absent, expired or caching is off."""
        ttl = self._ttl()
        if ttl <= 0:
            return None
        with self._lock:
            found = self._items.get(key)
            if found is None:
                return None
            stored_at, value = found
            if time.monotonic() - stored_at > ttl:
                del self._items[key]
                return None
            self._items.move_to_end(key)
        # A copy, because callers cut pages down in place: handing out the
        # stored dict would let one question's distillation become the next
        # question's page.
        return copy.deepcopy(value)

    def put(self, key: Hashable, value: Any) -> None:
        if self._ttl() <= 0:
            return
        with self._lock:
            self._items[key] = (time.monotonic(), copy.deepcopy(value))
            self._items.move_to_end(key)
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


# A page is at most WEB_READ_CHARS of text, so 200 of them is a few tens of MB
# in the worst case and nothing like that in practice.
PAGES = TTLCache(200)
# Results go stale faster than pages do: a quarter of the page lifetime.
SEARCHES = TTLCache(300, ttl_scale=0.25)
DISTILLED = TTLCache(500)


def clear() -> None:
    """Forget everything — for tests, and for anyone who wants a fresh look."""
    for cache in (PAGES, SEARCHES, DISTILLED):
        cache.clear()
