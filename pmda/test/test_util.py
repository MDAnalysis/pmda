# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
from __future__ import absolute_import

from six.moves import range, zip

import pytest

import time
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from pmda.util import timeit, make_balanced_slices


def test_timeit():
    with timeit() as timer:
        time.sleep(1)

    assert_almost_equal(timer.elapsed, 1, decimal=2)


@pytest.mark.parametrize("start", (None, 0, 1, 10))
@pytest.mark.parametrize("n_frames,n_blocks,result", [
    (5, 1, [slice(0, None, 1)]),
    (5, 2, [slice(0, 3, 1), slice(3, None, 1)]),
    (5, 3, [slice(0, 2, 1), slice(2, 4, 1), slice(4, None, 1)]),
    (5, 4, [slice(0, 2, 1), slice(2, 3, 1), slice(3, 4, 1),
            slice(4, None, 1)]),
    (5, 5, [slice(0, 1, 1), slice(1, 2, 1), slice(2, 3, 1), slice(3, 4, 1),
            slice(4, None, 1)]),
    (10, 2, [slice(0, 5, 1), slice(5, None, 1)]),
    (10, 3, [slice(0, 4, 1), slice(4, 7, 1), slice(7, None, 1)]),
    (10, 7, [slice(0, 2, 1), slice(2, 4, 1), slice(4, 6, 1), slice(6, 7, 1),
             slice(7, 8, 1), slice(8, 9, 1), slice(9, None, 1)]),
])
def test_make_balanced_slices_step1(n_frames, n_blocks, start, result, step=1):
    assert step in (None, 1), "This test can only test step None or 1"

    _start = start if start is not None else 0
    _result = [slice(sl.start + _start,
                     sl.stop + _start if sl.stop is not None else None,
                     sl.step) for sl in result]

    slices = make_balanced_slices(n_frames, n_blocks,
                                  sl=slice(start, None, step))
    assert_equal(slices, _result)


@pytest.mark.parametrize('n_blocks', [1, 2, 3, 4, 5, 7, 10, 11])
@pytest.mark.parametrize('start', [0, 1, 10])
@pytest.mark.parametrize('stop', [11, 20, 21])
@pytest.mark.parametrize('step', [None, 1, 2, 3, 5, 7])
@pytest.mark.parametrize('scale', [1, 2])
def test_make_balanced_slices(n_blocks, start, stop, step, scale):
    _start = start if start is not None else 0

    traj_frames = range(scale * stop)
    frames = traj_frames[start:stop:step]
    n_frames = len(frames)

    slices = make_balanced_slices(n_frames, n_blocks,
                                  sl=slice(start, stop, step))

    assert len(slices) == n_blocks

    # assemble frames again by blocks and show that we have all
    # the original frames

    block_frames = []
    for bslice in slices:
        block_frames.extend(list(traj_frames[bslice]))

    assert_equal(np.asarray(block_frames), np.asarray(frames))


@pytest.mark.parametrize('n_blocks', [1, 2])
@pytest.mark.parametrize('start', [0, 10])
@pytest.mark.parametrize('step', [None, 1, 2])
def test_make_balanced_slices_empty(n_blocks, start, step):
    slices = make_balanced_slices(0, n_blocks, sl=slice(start, None, step))
    assert slices == []


@pytest.mark.parametrize("n_frames,n_blocks,start",
                         [(-1, 5, None), (5, 0, None),
                          (5, -1, None), (0, 0, None), (-1, -1, None),
                          (5, 4, -1), (0, 5, -1), (5, 0, -1)])
def test_make_balanced_slices_ValueError(n_frames, n_blocks, start):
    with pytest.raises(ValueError):
        make_balanced_slices(n_frames, n_blocks, sl=slice(start, None))
