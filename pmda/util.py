# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
"""Utility functions --- :mod:`pmda.util`
=========================================


This module contains helper functions and classes that can be used throughout
:mod:`pmda`.

"""
from __future__ import absolute_import, division

import time

import numpy as np


class timeit(object):
    """measure time spend in context

    :class:`timeit` is a context manager (to be used with the :keyword:`with`
    statement) that records the execution time for the enclosed context block
    in :attr:`elapsed`.

    Attributes
    ----------
    elapsed : float
        Time in seconds that elapsed between entering
        and exiting the context.

    Example
    -------
    Use as a context manager::

       with timeit() as total:
          # code to be timed

       print(total.elapsed, "seconds")

    See Also
    --------
    :func:`time.time`

    """
    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        self.elapsed = end_time - self._start_time
        # always propagate exceptions forward
        return False


def make_balanced_slices(n_frames, n_blocks, start=None, stop=None, step=None):
    """Divide `n_frames` into `n_blocks` balanced blocks.

    The blocks are generated in such a way that they contain equal numbers of
    frames when possible, but there are also no empty blocks (which can happen
    with a naive distribution of ``ceil(n_frames/n_blocks)`` per block and a
    remainder block).

    If the trajectory is sliced in any way (``u.trajectory[start:stop:step]``)
    then the appropriate values for `start`, `stop`, and `step` must be passed
    to this function. Defaults can be set to ``None``. Only a subset of legal
    values for slices is supported: ``0 ≤ start ≤ stop`` and ``step ≥ 1``.

    Arguments
    ---------
    n_frames : int
        number of frames in the trajectory (≥0). This must be the
        number of frames *after* the trajectory has been sliced,
        i.e. ``len(u.trajectory[start:stop:step])``. If any of
        `start`, `stop, and `step` are not the defaults (left empty or
        set to ``None``) they must be provided as parameters.
    n_blocks : int
        number of blocks (>0)
    start : int or None
        The first index of the trajectory (default is ``None``, which
        is interpreted as "first frame", i.e., 0).
    stop : int or None
        The index of the last frame + 1 (default is ``None``, which is
        interpreted as "up to and including the last frame".
    step : int or None
        Step size by which the trajectory is sliced; the default is
        ``None`` which corresponds to ``step=1``.

    Returns
    -------
    slices : list of slice
        List of length ``n_blocks`` with one :class:`slice`
        for each block.

        If `n_frames` = 0 then an empty list ``[]`` is returned.

    Example
    -------
    For a trajectory with 5 frames and 4 blocks we get block sizes ``[2, 1, 1,
    1]`` (instead of ``[2, 2, 1, 0]`` with the naive algorithm).

    The slices will be ``[slice(0, 2, None), slice(2, 3, None),
    slice(3, 4, None), slice(4, 5, None)]``.

    The indices can be used to slice a trajectory into blocks::

        n_blocks = 5
        n_frames = len(u.trajectory[start:stop:step])

        slices = make_balanced_slices(n_frames, n_blocks,
                                      start=start, stop=stop, step=step)
        for i_block, block in enumerate(slices):
           for ts in u.trajectory[block]:
               # do stuff for block number i_block

    Notes
    -----
    Explanation of the algorithm: For `M` frames in the trajectory and
    `N` blocks (or processes), where `i` with 0 ≤ `i` ≤ `N` - 1 is the
    block number and `m[i]` is the number of frames for block `i` we
    get a *balanced distribution* (one that does not contain blocks of
    size 0) with the algorithm ::

        m[i] = M // N     # initial frames for block i
        r = M % N         # remaining frames 0 ≤ r < N
        for i in range(r):
            m[i] += 1     # distribute the remaining frames
                          # over the first r blocks

    For a `step` > 1, we use ``m[i] *= step``. The last slice will
    never go beyond the original `stop` if a value was provided.

    .. versionadded:: 0.2.0

    """

    start = int(start) if start is not None else 0
    stop = int(stop) if stop is not None else None
    step = int(step) if step is not None else 1

    if n_frames < 0:
        raise ValueError("n_frames must be >= 0")
    elif n_blocks < 1:
        raise ValueError("n_blocks must be > 0")
    elif start < 0:
        raise ValueError("start must be >= 0 or None")
    elif stop is not None and stop < start:
        raise ValueError("stop must be >= start and >= 0 or None")
    elif step < 1:
        raise ValueError("step must be > 0 or None")

    if n_frames == 0:
        # not very useful but allows calling code to work more gracefully
        return []

    bsizes = np.ones(n_blocks, dtype=np.int64) * n_frames // n_blocks
    bsizes += (np.arange(n_blocks, dtype=np.int64) < n_frames % n_blocks)
    # This can give a last index that is larger than the real last index;
    # this is not a problem for slicing but it's not pretty.
    # Example: original [0:20:3] -> n_frames=7, start=0, step=3:
    #          last frame 21 instead of 20
    bsizes *= step
    idx = np.cumsum(np.concatenate(([start], bsizes)))
    slices = [slice(bstart, bstop, step)
              for bstart, bstop in zip(idx[:-1], idx[1:])]

    # fix very last stop index: make sure it's within trajectory range or None
    # (no really critical because the slices will work regardless, but neater)
    last = slices[-1]
    last_stop = min(last.stop, stop) if stop is not None else stop
    slices[-1] = slice(last.start, last_stop, last.step)

    return slices
