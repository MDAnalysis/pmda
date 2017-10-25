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
from dask import distributed, multiprocessing
from dask.delayed import delayed
from joblib import cpu_count
import numpy as np

from .util import timeit


class Timing(object):
    """
    store various timeing results of obtained during a parallel analysis run
    """

    def __init__(self, io, compute, total, universe):
        self._io = io
        self._compute = compute
        self._total = total
        self._cumulate = np.sum(io) + np.sum(compute)
        self._universe = universe

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

    @property
    def universe(self):
        """time to create a universe for each block"""
        return self._universe


class ParallelAnalysisBase(object):
    """Base class for defining parallel multi frame analysis

    The class it is designed as a template for creating multiframe analyses.
    This class will automatically take care of setting up the trajectory
    reader for iterating in parallel.

    To define a new Analysis,
    :class:`~pmda.parallel.ParallelAnalysisBase` needs to be
    subclassed and
    :meth:`~pmda.parallel.ParallelAnalysisBase._single_frame` and
    :meth:`~pmda.parallel.ParallelAnalysisBase._conclude` must be
    defined. It is also possible to define
    :meth:`~~pmda.parallel.ParallelAnalysisBase._prepare` for
    pre-processing. See the example below.

    .. code-block:: python

       class NewAnalysis(ParallelAnalysisBase):
           def __init__(self, atomgroup, parameter):
               self._ag = atomgroup
               super(NewAnalysis, self).__init__(atomgroup.universe,
                                                 self._ag)

           def _single_frame(self, ts, agroups):
               # REQUIRED
               # called for every frame. ``ts`` contains the current time step
               # and ``agroups`` a tuple of atomgroups that are updated to the
               # current frame. Return result of `some_function` for a single
               # frame
               return some_function(agroups[0], self._parameter)

           def _conclude(self):
               # REQUIRED
               # Called once iteration on the trajectory is finished. Results
               # for each frame are stored in ``self._results`` in a per block
               # basis. Here those results should be moved and reshaped into a
               # sensible new variable.
               self.results = np.hstack(self._results)
               # Apply normalisation and averaging to results here if wanted.
               self.results /= np.sum(self.results)

    Afterwards the new analysis can be run like this.

    .. code-block:: python

       na = NewAnalysis(u.select_atoms('name CA'), 35).run()
       print(na.result)

    """

    def __init__(self, universe, atomgroups):
        """Parameters
        ----------
        Universe : :class:`~MDAnalysis.core.groups.Universe`
            a :class:`MDAnalysis.core.groups.Universe` (the
            `atomgroups` must belong to this Universe)

        atomgroups : tuple of :class:`~MDAnalysis.core.groups.AtomGroup`
            atomgroups that are iterated in parallel

        Attributes
        ----------
        _results : list
            The raw data from each process are stored as a list of
            lists, with each sublist containing the return values from
            :meth:`pmda.parallel.ParallelAnalysisBase._single_frame`.

        """
        self._trajectory = universe.trajectory
        self._top = universe.filename
        self._traj = universe.trajectory.filename
        self._indices = [ag.indices for ag in atomgroups]

    def _conclude(self):
        """Finalise the results you've gathered.

        Called at the end of the run() method to finish everything up.

        In general this method should unpack :attr:`self._results` to
        sensible variables.

        """
        pass

    def _prepare(self):
        """additional preparation to run"""
        pass

    def _single_frame(self, ts, atomgroups):
        """Perform computation on a single trajectory frame.

        Must return computed values as a list. You can only **read**
        from member variables stored in ``self``. Changing them during
        a run will result in undefined behavior. `ts` and any of the
        atomgroups can be changed (but changes will be overwritten
        when the next time step is read).

        Parameters
        ----------
        ts : :class:`~MDAnalysis.coordinates.base.Timestep`
            The current coordinate frame (time step) in the
            trajectory.
        atomgroups : tuple
            Tuple of :class:`~MDAnalysis.core.groups.AtomGroup`
            instances that are updated to the current frame.

        Returns
        -------
        values : anything
            The output from the computation over a single frame must
            be returned. The `value` will be added to a list for each
            block and the list of blocks is stored as :attr:`_results`
            before :meth:`_conclude` is run. In order to simplify
            processing, the `values` should be "simple" shallow data
            structures such as arrays or lists of numbers.

        """
        raise NotImplementedError

    def run(self,
            start=None,
            stop=None,
            step=None,
            scheduler=None,
            n_jobs=1,
            n_blocks=None):
        """Perform the calculation

        Parameters
        ----------
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        scheduler : dask scheduler, optional
            Use dask scheduler, defaults to multiprocessing. This can be used
            to spread work to a distributed scheduler
        n_jobs : int, optional
            number of jobs to start, if `-1` use number of logical cpu cores.
            This argument will be ignored when the distributed scheduler is
            used
        n_blocks : int, optional
            number of blocks to divide trajectory into. If ``None`` set equal
            to n_jobs or number of available workers in scheduler.

        """
        if scheduler is None:
            scheduler = multiprocessing

        if n_jobs == -1:
            n_jobs = cpu_count()

        if n_blocks is None:
            if scheduler == multiprocessing:
                n_blocks = n_jobs
            elif isinstance(scheduler, distributed.Client):
                n_blocks = len(scheduler.ncores())
            else:
                raise ValueError(
                    "Couldn't guess ideal number of blocks from scheduler."
                    "Please provide `n_blocks` in call to method.")

        scheduler_kwargs = {'get': scheduler.get}
        if scheduler == multiprocessing:
            scheduler_kwargs['num_workers'] = n_jobs

        start, stop, step = self._trajectory.check_slice_indices(
            start, stop, step)
        n_frames = len(range(start, stop, step))
        bsize = int(np.ceil(n_frames / float(n_blocks)))

        with timeit() as total:
            self._prepare()
            blocks = []
            for b in range(n_blocks):
                task = delayed(
                    self._dask_helper, pure=False)(
                        b * bsize * step + start,
                        min(stop, (b + 1) * bsize * step + start),
                        step,
                        self._indices,
                        self._top,
                        self._traj, )
                blocks.append(task)
            blocks = delayed(blocks)
            res = blocks.compute(**scheduler_kwargs)
            self._results = np.asarray([el[0] for el in res])
            self._conclude()

        self.timing = Timing(
            np.hstack([el[1] for el in res]),
            np.hstack([el[2] for el in res]), total.elapsed,
            np.array([el[3] for el in res]))
        return self

    def _dask_helper(self, start, stop, step, indices, top, traj):
        """helper function to actually setup dask graph"""
        with timeit() as b_universe:
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

        return np.asarray(res), np.asarray(times_io), np.asarray(
            times_compute), b_universe.elapsed
