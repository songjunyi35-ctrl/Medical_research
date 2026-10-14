from concurrent.futures import ThreadPoolExecutor
import time
import pytest
from sammed3d.bounded_ordered_pipeline import bounded_ordered_map


def test_order_and_bound():
    class Executor:
        outstanding=0
        maximum=0
        def submit(self,fn,item):
            self.outstanding+=1;self.maximum=max(self.maximum,self.outstanding)
            owner=self
            class Future:
                def result(self,timeout):
                    owner.outstanding-=1
                    return fn(item)
                def cancel(self):pass
            return Future()
    ex=Executor()
    assert list(bounded_ordered_map(ex,lambda x:x*x,range(17),4,time.monotonic()+5))==[x*x for x in range(17)]
    assert ex.maximum==4 and ex.outstanding==0


def test_real_executor_order():
    with ThreadPoolExecutor(max_workers=3) as ex:
        assert list(bounded_ordered_map(ex,lambda x:2*x,range(10),4,time.monotonic()+5))==list(range(0,20,2))


def test_worker_failure_propagates():
    def fail(x):raise RuntimeError('worker failed')
    with ThreadPoolExecutor(max_workers=1) as ex:
        with pytest.raises(RuntimeError,match='worker failed'):
            list(bounded_ordered_map(ex,fail,[1,2],2,time.monotonic()+5))


def test_expired_deadline_submits_nothing():
    class Forbidden:
        def submit(self,*args):raise AssertionError('should not submit')
    with pytest.raises(TimeoutError):
        list(bounded_ordered_map(Forbidden(),lambda x:x,[1],1,time.monotonic()-1))


@pytest.mark.parametrize('bound',[0,-1,True,1.5])
def test_invalid_bound(bound):
    with ThreadPoolExecutor(max_workers=1) as ex:
        with pytest.raises(ValueError):
            list(bounded_ordered_map(ex,lambda x:x,[1],bound,time.monotonic()+5))
