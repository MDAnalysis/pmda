# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
"""
Utility functions --- :mod:`pmda.util`
=====================================================================


"""
from __future__ import absolute_import, division

import time

import numpy as np


class timeit(object):
    """measure time spend in context

    """
    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        self.elapsed = end_time - self._start_time
        # always propagate exceptions forward
        return False


def make_balanced_blocks(n_frames, n_blocks, start=None, step=None):
    """Divide `n_frames` into `n_blocks` balanced blocks.

    The blocks are generated in such a way that they contain equal numbers of
    frames when possible, but there are also no empty blocks (which can happen
    with a naive distribution of ``ceil(n_frames/n_blocks)`` per block and a
    remainder block).

    Arguments
    ---------
    n_frames : int
        number of frames in the trajectory (>0)
    n_blocks : int
        number of blocks (>0)
    start : int
        first index of the trajectory (default is None, which is
        interpreted as "first frame", i.e., 0)

    Returns
    -------
    frame_indices : array
        Array of shape of length ``n_blocks + 1`` with the starting frame of
        each block; the last index ``frame_indices[-1]`` corresponds to the
        last index of the last block + 1 so that one can easily use the array
        for slicing, as shown in the example below.

    Example
    -------
    For a trajectory with 5 frames and 4 blocks we get block sizes ``[2, 1, 1,
    1]`` (instead of ``[2, 2, 1, 0]``).

    The indices will be ``[0, 2, 3, 4, 5]``.

    The indices can be used to slice a trajectory into blocks::

        idx = assign_blocks(5, 4)
        for i_block, (start, stop) in enumerate(zip(idx[:-1], idx[1:])):
           for ts in u.trajectory[start:stop]:
               # do stuff for block number i_block


    Notes
    -----
    Explanation of the algorithm: For `M` trajectory frames in the trajectory
    and `N` blocks (or processes), where `i` with 0 ≤ i N-1 is the block number
    and `m[i]` is the number of frames for block `i` we get a *balanced
    distribution* (one that does not contain blocks of size 0) with the
    algorithm ::

        m[i] = M // N     # initial frames for block i
        r = M % N         # remaining frames 0 ≤ r < N
        for i in range(r):
            m[i] += 1     # distribute the remaining frames
                          # over the first r blocks


    .. versionadded:: 0.2.0

    """
    start = start if start is not None else 0
    step = step if step is not None else 1

    if n_frames <= 0:
        raise ValueError("n_frames must be > 0")
    elif n_blocks <= 0:
        raise ValueError("n_blocks must be > 0")
    elif start < 0:
        raise ValueError("start must be >= 0")
    if step != 1:
        raise NotImplementedError("Only step=1 or step=None is supported")
    # TODO: use step

    bsizes = np.ones(n_blocks, dtype=np.int64) * n_frames // n_blocks
    bsizes += (np.arange(n_blocks, dtype=np.int64) < n_frames % n_blocks)
    frame_indices = np.cumsum(np.concatenate(([start], bsizes)))
    return frame_indices
