# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
"""
Parallel Analysis building blocks --- :mod:`pmda.parallel`
==========================================================

A collection of useful building blocks for creating Analysis
classes.

"""
from __future__ import absolute_import, division
from six.moves import range

import MDAnalysis as mda
from dask.delayed import delayed
from joblib import cpu_count
import numpy as np

from .util import timeit


class Timing(object):
    """
    store various timeing results of obtained during a parallel analysis run
    """
    def __init__(self, io, compute, total):
        self._io = io
        self._compute = compute
        self._total = total
        self._cumulate = np.sum(io) + np.sum(compute)

    @property
    def io(self):
        """io time per frame"""
        return self._io

    @property
    def compute(self):
        """compute time per frame"""
        return self._compute

    @property
    def total(self):
        """wall time"""
        return self._total

    @property
    def cumulate_time(self):
        """cumulative time of io and compute for each frame. This isn't equal to
        `self.total / n_jobs` because `self.total` also includes the scheduler
        overhead

        """
        return self._cumulate


class ParallelAnalysisBase(object):
    """Base class for defining parallel multi frame analysis

    The class it is designed as a template for creating multiframe analyses.
    This class will automatically take care of setting up the trajectory
    reader for iterating, and it offers to show a progress meter.

    """

    def __init__(self, universe, atomgroups):
        """
        Parameters
        ----------
        Universe : mda.Universe
            A Universe
        atomgroups : array of AtomGroup
            atomgroups that are iterated in parallel
        """
        self._trajectory = universe.trajectory
        self._top = universe.filename
        self._traj = universe.trajectory.filename
        self._indices = [ag.indices for ag in atomgroups]

    def _conclude(self):
        """Finalise the results you've gathered.

        Called at the end of the run() method to finish everything up.

        In general this method should unpack `self._results` to sensible
        variables

        """
        pass

    def _prepare(self):
        """additional preparation to run"""
        pass

    def _single_frame(self, ts, atomgroups):
        """must return computed values"""
        raise NotImplementedError

    def run(self, n_jobs=1, start=None, stop=None, step=None, get=None):
        """Perform the calculation

        Parameters
        ----------
        n_jobs : int, optional
            number of jobs to start, if `-1` use number of logical cpu cores
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        get : scheduler, optional
            dask or distributed scheduler; by default, the dask
            default is used
        """
        if n_jobs == -1:
            n_jobs = cpu_count()

        start, stop, step = self._trajectory.check_slice_indices(
            start, stop, step)
        n_frames = len(range(start, stop, step))

        n_blocks = n_jobs
        bsize = int(np.ceil(n_frames / float(n_blocks)))

        with timeit() as total:
            blocks = []
            for b in range(n_blocks):
                task = delayed(
                    self.dask_helper, pure=False)(
                        b * bsize + start,
                        (b + 1) * bsize * step,
                        step,
                        self._indices,
                        self._top,
                        self._traj, )
                blocks.append(task)
            blocks = delayed(blocks)
            res = blocks.compute(get=get)
            self._results = np.asarray([el[0] for el in res])
            self._conclude()

        self.timing = Timing(
            np.hstack([el[1] for el in res]),
            np.hstack([el[2] for el in res]), total.elapsed)
        return self

    def dask_helper(self, start, stop, step, indices, top, traj):
        """helper function to actually setup dask graph"""
        u = mda.Universe(top, traj)
        agroups = [u.atoms[idx] for idx in indices]

        res = []
        times_io = []
        times_compute = []
        for i in range(start, stop, step):
            with timeit() as b_io:
                ts = u.trajectory[i]
            with timeit() as b_compute:
                res.append(self._single_frame(ts, agroups))
            times_io.append(b_io.elapsed)
            times_compute.append(b_compute.elapsed)

        return np.asarray(res), np.asarray(times_io), np.asarray(times_compute)
