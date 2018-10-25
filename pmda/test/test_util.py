# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
from __future__ import absolute_import

import pytest

import time
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pmda.util import timeit, make_balanced_blocks


def test_timeit():
    with timeit() as timer:
        time.sleep(1)

    assert_almost_equal(timer.elapsed, 1, decimal=2)

@pytest.mark.parametrize("n_frames,n_blocks,result", [
    (5, 1, [0, 5]),
    (5, 2, [0, 3, 5]),
    (5, 3, [0, 2, 4, 5]),
    (5, 4, [0, 2, 3, 4 ,5]),
    (5, 5, [0, 1, 2, 3, 4 ,5]),
    (10, 2, [0, 5, 10]),
    (10, 3, [0, 4, 7, 10]),
    (10, 7, [0, 2, 4, 6, 7, 8, 9, 10]),
])
@pytest.mark.parametrize("start", (None, 0, 1, 10))
def test_make_balanced_blocks(n_frames, n_blocks, start, result):
    start = start if start is not None else 0
    result = np.array(result) + start

    idx = make_balanced_blocks(n_frames, n_blocks, start=start)
    assert_equal(idx, result)

@pytest.mark.parametrize("n_frames,n_blocks,start",
                         [(0, 5, None), (-1, 5, None), (5, 0, None),
                          (5, -1, None), (0, 0, None), (-1, -1, None),
                          (5, 4, -1), (0, 5, -1), (5, 0, -1)])
def test_make_balanced_blocks_ValueError(n_frames, n_blocks, start):
    with pytest.raises(ValueError):
        make_balanced_blocks(n_frames, n_blocks, start=start)

