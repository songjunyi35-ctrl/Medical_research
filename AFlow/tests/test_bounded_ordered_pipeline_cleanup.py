"""Failure-injection checks; no processes, patient files or GPU involved."""
import time
import pytest
from sammed3d.bounded_ordered_pipeline import bounded_ordered_map


class Future:
    def __init__(self,item,error=None):
        self.item=item;self.error=error;self.cancelled=False;self.read=False

    def result(self,timeout):
        assert timeout>0
        self.read=True
        if self.error:raise self.error
        return self.item

    def cancel(self):
        self.cancelled=True


class Executor:
    def __init__(self,error=None,submit_error_at=None):
        self.futures=[];self.error=error;self.submit_error_at=submit_error_at

    def submit(self,fn,item):
        if len(self.futures)==self.submit_error_at:raise RuntimeError('submit failed')
        future=Future(item,self.error if not self.futures else None)
        self.futures.append(future)
        return future


def test_consumer_close_cancels_unconsumed_futures():
    ex=Executor()
    values=bounded_ordered_map(ex,lambda x:x,range(100),4,time.monotonic()+10)
    assert next(values)==0 and len(ex.futures)==4
    values.close()
    assert not ex.futures[0].cancelled
    assert all(f.cancelled for f in ex.futures[1:])


@pytest.mark.parametrize('error',[RuntimeError('worker failed'),TimeoutError('future timeout')])
def test_failure_cancels_all_pending(error):
    ex=Executor(error=error)
    with pytest.raises(type(error)):
        list(bounded_ordered_map(ex,lambda x:x,range(100),4,time.monotonic()+10))
    assert len(ex.futures)==4 and all(f.cancelled for f in ex.futures)
    assert sum(f.read for f in ex.futures)==1


def test_submission_failure_cancels_already_queued():
    ex=Executor(submit_error_at=2)
    with pytest.raises(RuntimeError,match='submit failed'):
        list(bounded_ordered_map(ex,lambda x:x,range(100),4,time.monotonic()+10))
    assert len(ex.futures)==2 and all(f.cancelled for f in ex.futures)


def test_fully_consumed_stream_does_not_cancel_completed_results():
    ex=Executor()
    assert list(bounded_ordered_map(ex,lambda x:x,range(25),8,time.monotonic()+10))==list(range(25))
    assert len(ex.futures)==25 and all(f.read and not f.cancelled for f in ex.futures)
