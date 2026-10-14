"""Executor-agnostic ordered map with a strict unconsumed task bound."""
from collections import deque
import time


def bounded_ordered_map(executor, function, items, max_pending, deadline):
    if type(max_pending) is not int or max_pending < 1:
        raise ValueError('positive integer max_pending required')
    pending = deque()
    iterator = iter(items)
    exhausted = False
    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < max_pending:
                if time.monotonic() >= deadline:
                    raise TimeoutError('pipeline deadline expired')
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(executor.submit(function,item))
            if pending:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('pipeline deadline expired')
                # Keep the future in the queue until success, so finally also cancels it.
                result = pending[0].result(timeout=remaining)
                pending.popleft()
                yield result
    finally:
        for future in pending:
            future.cancel()
