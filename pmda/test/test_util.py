# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
from __future__ import absolute_import

from six.moves import range

import pytest

import time
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from functools import reduce

from pmda.util import timeit, make_balanced_slices, second_order_moments, sumofsquares


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
                                  start=start, step=step)
    assert_equal(slices, _result)


def _test_make_balanced_slices(n_blocks, start, stop, step, scale):
    _start = start if start is not None else 0

    traj_frames = range(scale * stop)
    frames = traj_frames[start:stop:step]
    n_frames = len(frames)

    slices = make_balanced_slices(n_frames, n_blocks,
                                  start=start, stop=stop, step=step)

    assert len(slices) == n_blocks

    # assemble frames again by blocks and show that we have all
    # the original frames; get the sizes of the blocks

    block_frames = []
    block_sizes = []
    for bslice in slices:
        bframes = traj_frames[bslice]
        block_frames.extend(list(bframes))
        block_sizes.append(len(bframes))
    block_sizes = np.array(block_sizes)

    # check that we have all the frames accounted for
    assert_equal(np.asarray(block_frames), np.asarray(frames))

    # check that the distribution is balanced
    if n_frames >= n_blocks:
        assert np.all(block_sizes > 0)
        minsize = n_frames // n_blocks
        assert len(np.setdiff1d(block_sizes, [minsize, minsize+1])) == 0, \
            "For n_blocks <= n_frames, block sizes are not balanced"
    else:
        # pathological case; we will have blocks with length 0
        # and n_blocks with 1 frame
        zero_blocks = block_sizes == 0
        assert np.sum(zero_blocks) == n_blocks - n_frames
        assert np.sum(~zero_blocks) == n_frames
        assert len(np.setdiff1d(block_sizes[~zero_blocks], [1])) == 0, \
            "For n_blocks>n_frames, some blocks contain != 1 frame"


@pytest.mark.parametrize('n_blocks', [1, 2, 3, 4, 5, 7, 10, 11])
@pytest.mark.parametrize('start', [0, 1, 10])
@pytest.mark.parametrize('stop', [11, 100, 256])
@pytest.mark.parametrize('step', [None, 1, 2, 3, 5, 7])
@pytest.mark.parametrize('scale', [1, 2])
def test_make_balanced_slices(n_blocks, start, stop, step, scale):
    return _test_make_balanced_slices(n_blocks, start, stop, step, scale)


def test_make_balanced_slices_step_gt_stop(n_blocks=2, start=None,
                                           stop=5, step=6, scale=1):
    return _test_make_balanced_slices(n_blocks, start, stop, step, scale)


@pytest.mark.parametrize('n_blocks', [1, 2])
@pytest.mark.parametrize('start', [0, 10])
@pytest.mark.parametrize('step', [None, 1, 2])
def test_make_balanced_slices_empty(n_blocks, start, step):
    slices = make_balanced_slices(0, n_blocks, start=start, step=step)
    assert slices == []


@pytest.mark.parametrize("n_frames,n_blocks,start,stop,step",
                         [(-1, 5, None, None, None), (5, 0, None, None, None),
                          (5, -1, None, None, None), (0, 0, None, None, None),
                          (-1, -1, None, None, None),
                          (5, 4, -1, None, None), (0, 5, -1, None, None),
                          (5, 0, -1, None, None),
                          (5, 4, None, -1, None), (5, 4, 3, 2, None),
                          (5, 4, None, None, -1), (5, 4, None, None, 0)])
def test_make_balanced_slices_ValueError(n_frames, n_blocks,
                                         start, stop, step):
    with pytest.raises(ValueError):
        make_balanced_slices(n_frames, n_blocks,
                             start=start, stop=stop, step=step)


@pytest.mark.parametrize('n_frames', [3, 4, 10, 19, 101, 331, 1000])
def test_second_order_moments(n_frames):
    # generate array of random positions in range [-100, 100] for 100 time steps
    pos = 200*(np.random.random(size=(n_frames, 1000, 3)) - 0.5)
    # random splitting point
    isplit = (np.random.randint(1, n_frames-1))
    # split into two partitions; should be fixed to split by n_blocks partitions
    p1, p2 = pos[:isplit], pos[isplit:]
    # create [t, mu, M] lists
    S1 = [len(p1), p1.mean(axis=0), sumofsquares(p1)]
    S2 = [len(p2), p2.mean(axis=0), sumofsquares(p2)]
    # run lists through second_order_moments
    result = second_order_moments(S1, S2)
    # compare result to calculations over entire pos array
    assert result[0] == len(pos)
    assert_almost_equal(result[1], pos.mean(axis=0))
    assert_almost_equal(result[2], sumofsquares(pos))


@pytest.mark.parametrize('n_frames', [1000, 10000, 50000, 100000])
@pytest.mark.parametrize('n_blocks', [2, 3, 4, 5, 10, 100, 500])
def test_reduce(n_frames, n_blocks):
    # generate array of random positions in range [-100, 100] for 100 time steps
    pos = 200*(np.random.random(size=(n_frames, 1000, 3)) - 0.5).astype(np.float64)
    # generate array of indices to split pos by
    split = []
    for i in range(n_blocks):
        split.append(np.random.randint(1, n_frames-1))
    # remove reduntant indices and sort
    split = np.sort(np.unique(split))
    # generate list of block slices
    split_pos = []
    for i in range(len(split)):
        if i == 0:
            split_pos.append(pos[0:split[i]])
        elif not i == 0 and not i == len(split)-1:
            split_pos.append(pos[split[i-1]:split[i]])
        else:
            split_pos.append(pos[split[-2]:split[-1]])
            split_pos.append(pos[split[-1]:n_frames])
    # create list of [t, mu, M] lists
    S = []
    for i in range(len(split_pos)):
        S.append([len(split_pos[i]), split_pos[i].mean(axis=0, dtype=np.float64), sumofsquares(split_pos[i])])
    # combine block results using fold method
    results = reduce(second_order_moments, S)
    # compare result to calculations over entire pos array
    assert results[0] == len(pos)
    # check that the mean of the original pos array is equal to the collected mean array from reduce()
    assert_almost_equal(results[1], pos.mean(axis=0, dtype=np.float64))
    # check that the sum of square arrays are equal
    # Note: 'decimal' was changed from the default '7' to '5' because the
    # absolute error for large trajectory lengths (n_frames > 1e4) is not
    # almost equal to 7 decimal places
    assert_almost_equal(results[2], sumofsquares(pos), decimal=5)
