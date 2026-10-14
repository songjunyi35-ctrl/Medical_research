"""Run UTSW Amendment 01 with only bounded transport retries/concurrency."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor as OriginalPool

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scripts import audit_utsw_glioma_source_amendment_01 as audit


OriginalSession = requests.Session


def resilient_session() -> requests.Session:
    session = OriginalSession()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        allowed_methods=frozenset(("GET", "POST")),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class LimitedPool(OriginalPool):
    def __init__(self, max_workers=None, *args, **kwargs):
        requested = 4 if max_workers is None else int(max_workers)
        super().__init__(max_workers=min(requested, 4), *args, **kwargs)


def main() -> None:
    audit.requests.Session = resilient_session
    audit.ThreadPoolExecutor = LimitedPool
    audit.main()


if __name__ == "__main__":
    main()
