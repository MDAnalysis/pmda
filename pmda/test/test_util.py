import time
from numpy.testing import assert_almost_equal

from pmda.util import timeit


def test_timeit():
    with timeit() as timer:
        time.sleep(1)

    assert_almost_equal(timer.elapsed, 1, decimal=3)
