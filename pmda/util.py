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


def make_balanced_blocks(n_frames, n_blocks, start=None, step=None):
    """Divide `n_frames` into `n_blocks` balanced blocks.

    The blocks are generated in such a way that they contain equal numbers of
    frames when possible, but there are also no empty blocks (which can happen
    with a naive distribution of ``ceil(n_frames/n_blocks)`` per block and a
    remainder block).

    Arguments
    ---------
    n_frames : int
        number of frames in the trajectory (≥0). This must be the number of
        frames *after* the trajectory has been sliced,
        i.e. ``len(u.trajectory[start:stop:step])`` and `start` and `step` must
        be provided as parameters.
    n_blocks : int
        number of blocks (>0)
    start : int
        first index of the trajectory (default is ``None``, which is
        interpreted as "first frame", i.e., 0); see comments on `n_frames`.
    step : int
        step by which the trajectory is sliced (see description of `n_frames`);
        the default is ``None`` which corresponds to ``step=1``.

    Returns
    -------
    frame_indices : array
        Array of shape of length ``n_blocks + 1`` with the starting frame of
        each block; the last index ``frame_indices[-1]`` corresponds to the
        last index of the last block + 1 so that one can easily use the array
        for slicing, as shown in the example below.

        If `n_frames` = 0 then an empty list ``[]`` is returned.

        .. note:: For `step` > 1 the last index in `frame_indices` may be
                  larger than the real last index in the trajectory; this is
                  not a problem for slicing but it means that one should *not*
                  rely on the last index for any other operations except
                  slicing of the original trajectory.

    Example
    -------
    For a trajectory with 5 frames and 4 blocks we get block sizes ``[2, 1, 1,
    1]`` (instead of ``[2, 2, 1, 0]``).

    The indices will be ``[0, 2, 3, 4, 5]``.

    The indices can be used to slice a trajectory into blocks::

        n_blocks = 5
        n_frames = len(u.trajectory[start:stop:step])

        idx = assign_blocks(n_frames, n_blocks)
        for i_block, (bstart, bstop) in enumerate(zip(idx[:-1], idx[1:])):
           for ts in u.trajectory[bstart:bstop:step]:
               # do stuff for block number i_block

    Note that in order to access the correct frames in each block, the
    trajectory *must be sliced with the original `step` value* (``step`` in the
    example) and the start and stop indices returned by
    :func:`make_balanced_blocks`, namely ``bstart`` and ``bstop`` in the
    example above.


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

    For a `step` > 1, we use ``m[i] *= step``. This approach can give a last
    index that is larger than the real last index; this is not a problem for
    slicing but it's not pretty. As an example, we have the original trajectory
    slice ``[0:20:3]``, which corresponds to ``n_frames=7``, ``start=0``,
    ``step=3`` and the last frame index will always be 21 instead of 20 (for
    instance for ``n_blocks=1`` we get ``[0, 21]`` and for two blocks, ``[0,
    12, 21]``).


    .. versionadded:: 0.2.0

    """
    start = start if start is not None else 0
    step = step if step is not None else 1

    if n_frames < 0:
        raise ValueError("n_frames must be >= 0")
    elif n_blocks <= 0:
        raise ValueError("n_blocks must be > 0")
    elif start < 0:
        raise ValueError("start must be >= 0")

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
    frame_indices = np.cumsum(np.concatenate(([start], bsizes)))
    return frame_indices
