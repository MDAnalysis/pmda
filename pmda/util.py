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

import functools

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


def second_order_moments(S1, S2):
    r"""Calculates the combined second order moment.

    Given the partial centered moments of two partitions (S1 and S2) of a data
    set S, calculates the second order moments of S = S1 ∪ S2.

    Parameters
    ----------
    S1 : array
       Contains `(T1, mu1, M1)` where `T1` is an integer (number of elements
       in the partition, e.g., the number of time frames), `mu1` is an
       `n x m` array of the means for `n` atoms (and for example, `m=3` for
       the center of geometry), `M1` is also an `n x m` array of the sum of
       squares.
    S2 : array
       Contains `(T2, mu2, M2)` where `T2` is an integer (number of elements
       in the partition, e.g., the number of time frames), `mu2` is an
       `n x m` array of the means for `n` atoms (and for example, `m=3` for
       the center of geometry), `M2` is also an `n x m` array of the sum of
       squares.

    Returns
    -------
    S : (T, mu, M)
       The returned tuple contains the total number of elements in the
       partition `T`, the mean `mu` and the "second moment" `M` (sum of
       squares) for the combined data.

    Notes
    -----
    For a given set of data, the sum of squares, also known as the second
    order moment about the mean, is defined as the sum of the squares of the
    differences between each data point and the mean (sum of the individual
    deviations from the mean) of the data set,

    .. math::

        M_{2, S_{i}} = \sum_{t=t_{0}}^{T}(x_{t} - \mu_{i})^2,

    where :math:`\mu_{i}` is the time average of :math:`x_{t}`. If the
    average of the squares of the individual deviations is taken (instead of
    the sum), this yields the variance:

    .. math::

        \sigma_{i}^{2} = \frac{1}{T}\sum_{t=t_{0}}^{T}(x_{t} - \mu_{i})^2

    In order to combine the mean and second order moments of two separate
    partitions, [CGL1979]_ derived the following formulae:

    .. math::

        \mu = \frac{T_{1}\mu_{1} + T_{2}\mu_{2}}{T}

    and

    .. math::

        M_{2, S} = M_{2, S_{1}} + M_{2, S_{2}} + \
        \frac{T_{1}T_{2}}{T}(\mu_{2} - \mu_{1})^{2},

    where :math:`T`, :math:`T_{1}`, and :math:`T_{2}` are the respective
    cardinalities of :math:`S`, :math:`S_{1}`, and :math:`S_{2}`, :math:`\mu`,
    :math:`\mu_{1}`, and :math:`\mu_{2}` are the respective means of
    :math:`S`, :math:`S_{1}`, and :math:`S_{2}`, and :math:`M_{2, S}`,
    :math:`M_{2, S_{1}}`, and :math:`M_{2, S_{2}}` are the respective second
    order moments of :math:`S`, :math:`S_{1}`, and :math:`S_{2}`. This is
    similar notation to [Pebay2008]_. With a combined sum of squares and mean,
    it is possible to calculate the root-mean-square fluctuations, otherwise
    known as the population standard deviation:

    .. math::

        \sigma_{i} = \sqrt{\frac{1}{T} \
        \sum_{t=t_{0}}^{T}(x_{t} - \mu_{i})^2}

    References
    ----------
    .. [CGL1979] T. F. Chan, G. H. Golub, and R. J. LeVeque. "Updating
       formulae and a pairwise algorithm for computing sample variances."
       Technical Report STAN-CS-79-773, Stanford University, Department of
       Computer Science, 1979.
    .. [Pebay2008] P. Pebay. "Formulas for robust one-pass parallel
       computation of co-variances and arbitrary-order statistical moments."
       Technical Report SAND2008-6212, 2008.


    .. versionadded:: 0.3.0
    """

    T = S1[0] + S2[0]
    mu = (S1[0]*S1[1] + S2[0]*S2[1])/T
    M = S1[2] + S2[2] + (S1[0] * S2[0]/T) * (S2[1] - S1[1])**2
    S = T, mu, M

    return S


def fold_second_order_moments(*args):
    """Fold action for :func:`second_order_moments` calculation.

    Takes in data that can be combined associatively (order doesn't matter) and
    applies a combining function in a recursive fashion. In this case, it takes
    in a list of lists that each contain the total number of time steps, an `n
    x m` array of mean positions for `n` atoms, and an `n x m` array of second
    order moments for `n` atoms, for a given partition of a trajectory. It
    takes the first partition, combines it with the second, combines that
    result with the third, and that result with the fourth, etc. using
    :func:`second_order_moments`. The final result is a list of the summed time
    steps, combined mean positions, and combined second order moments of all
    atoms in the combined trajectory.

    See Also
    --------
    `Haskell fold/reduce method <https://wiki.haskell.org/Fold>`_


    .. versionadded:: 0.3.0
    """
    return functools.reduce(second_order_moments, *args)
