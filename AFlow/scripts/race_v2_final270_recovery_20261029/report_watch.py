"""Wait for both complete worker audits, then report actual reconstruction cost."""
import time
from .common import R,ROOT,SELECTION_SHA,guard,read,sha,write
from scripts.race_v2_stage_20261029.published_raw import verified
from . import report


def main():
    frozen=read(R/'PUBLICATION_FREEZE.json')
    for path,digest in frozen['source_sha256'].items():
        assert sha(ROOT/path)==digest,path
    while True:
        guard()
        if verified(R/'a/RAW_SINGLE.json',SELECTION_SHA) is not None and verified(R/'b/RAW_FUSION.json',SELECTION_SHA) is not None:
            report.complete()
            return
        time.sleep(15)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        write(R/'PUBLICATION_FAILURE.json',dict(at_unix=time.time(),error=repr(error),no_patient_neural_retry=True))
        raise
