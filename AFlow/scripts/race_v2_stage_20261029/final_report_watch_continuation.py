"""Wait for fully audited published wrappers, not a bare raw-report pathname."""
from __future__ import annotations

import json
import time

from .governance import RECORD, ROOT, atomic_json, guard, sha
from . import final_report_continuation as final_report


def ready(path):
    from .published_raw import verified
    return verified(path) is not None


def main():
    frozen = json.loads((RECORD/'FINAL_PUBLICATION_CONTINUATION_FREEZE.json').read_text())
    for path, digest in frozen['source_sha256'].items():
        assert sha(ROOT/path) == digest, path
    while True:
        guard(1154121)
        if ((RECORD/'FINAL_TEST_SELECTION.json').exists()
                and ready(final_report.A/'final270/RAW_SINGLE.json')
                and ready(final_report.B/'final270/RAW_FUSION.json')):
            final_report.complete()
            return
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        atomic_json(RECORD/'FINAL_PUBLICATION_CONTINUATION_FAILURE.json', {'at_unix': time.time(), 'error': repr(error),
                    'saved_predictions_retained': True, 'no_automatic_neural_retry': True})
        raise
