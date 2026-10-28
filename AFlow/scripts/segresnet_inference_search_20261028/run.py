"""Exclusive, finite supervisor; no training/test entry points and no silent retries."""
import fcntl
import subprocess
import traceback
from scripts.segresnet_inference_search_20261028.common import *


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    lock = (R / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify()
    if not (R / 'RUN_STARTED.json').exists():
        write(R / 'RUN_STARTED.json', dict(at=now(), unix=time.time(), worker=1152682, pid=os.getpid()))
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLCONFIGDIR=str(TMP / 'mpl'))
    for module, done in [('preflight', 'PREFLIGHT.json'), ('atoms', 'ATOMS_DONE.json'),
                         ('search', 'STAGE1_DONE.json'), ('postprocess', 'STAGE2_DONE.json'),
                         ('summarize', 'ARCHIVE_SHA256.json')]:
        if (R / done).exists():
            continue
        guard()
        status('starting_' + module)
        with (TMP / (module + '.log')).open('ab') as log:
            process = subprocess.run([str(PY), '-m', 'scripts.segresnet_inference_search_20261028.' + module],
                                     cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=max(1, read(R / 'AUTHORIZATION.json')['deadline_unix'] - time.time()))
        if process.returncode:
            raise RuntimeError(f'{module} failed with exit_code {process.returncode}; see worker scratch log')
        if module == 'atoms':
            write(R / done, dict(at=now(), status='complete'))
    status('complete', result='RESULT.json', report='RESULT.md')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        failure = dict(at=now(), error=repr(error), traceback=traceback.format_exc(),
                       training=False, test_access=False)
        write(R / f'FAILURE_{time.time_ns()}.json', failure)
        status('failed', **{'failure': failure})
        raise
