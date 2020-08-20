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
from contextlib import contextmanager
from joblib import cpu_count
import time
import warnings
import uuid

import MDAnalysis as mda
import dask
from dask.base import DaskMethodsMixin
import dask.distributed
from dask.utils import funcname
from dask.base import tokenize
import numpy as np

from .util import timeit, make_balanced_slices


class Timing(object):
    """
    store various timeing results of obtained during a parallel analysis run
    """

    def __init__(self, io, compute, total, prepare, prepare_dask,
                 conclude, wait=None, io_block=None,
                 compute_block=None):
        self._io = io
        self._io_block = io_block
        self._compute = compute
        self._compute_block = compute_block
        self._total = total
        self._cumulate = np.sum(io) + np.sum(compute)
        self._prepare = prepare
        self._prepare_dask = prepare_dask
        self._conclude = conclude
        self._wait = wait

    @property
    def io(self):
        """io time per frame"""
        return self._io

    @property
    def io_block(self):
        """io time per block"""
        return self._io_block

    @property
    def compute(self):
        """compute time per frame"""
        return self._compute

    @property
    def compute_block(self):
        """compute time per block"""
        return self._compute_block

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
    def prepare(self):
        """time to prepare"""
        return self._prepare

    @property
    def prepare_dask(self):
        """time to submit jobs to dask"""
        return self._prepare_dask

    @property
    def conclude(self):
        """time to conclude"""
        return self._conclude

    @property
    def wait(self):
        """time for blocks to start working"""
        return self._wait


class ParallelAnalysisBase(DaskMethodsMixin):
    """Base class for defining parallel multi frame analysis

    The class it is designed as a template for creating multiframe analyses.
    This class will automatically take care of setting up the trajectory
    reader for iterating in parallel.

    To parallelize the analysis ``ParallelAnalysisBase`` separates the
    trajectory into work blocks containing multiple frames. The number of
    blocks is equal to the number of available cores or dask workers. This
    minimizes the number of python processes that are started during a
    calculation. Accumulation of frames within a block happens in the
    `self._reduce` function. A consequence when using dask is that adding
    additional workers during a computation will not result in an reduction
    of run-time.


    To define a new Analysis,
    :class:`~pmda.parallel.ParallelAnalysisBase` needs to be
    subclassed and
    :meth:`~pmda.parallel.ParallelAnalysisBase._single_frame` and
    :meth:`~pmda.parallel.ParallelAnalysisBase._conclude` must be
    defined. It is also possible to define
    :meth:`~~pmda.parallel.ParallelAnalysisBase._prepare` for
    pre-processing and :meth:`~~pmda.parallel.ParallelAnalysisBase._reduce`
    for custom reduce operation on the work blocks. See the example below.

    .. code-block:: python

       class NewAnalysis(ParallelAnalysisBase):
           def __init__(self, atomgroup, parameter):
               self._ag = atomgroup
               super().__init__(atomgroup.universe,
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
               self.results /= np.sum(self.results

           @staticmethod
           def _reduce(res, result_single_frame):
               # NOT REQUIRED
               # Called for every frame. ``res`` contains all the results
               # before current time step, and ``result_single_frame`` is the
               # result of self._single_frame for the current time step. The
               # return value is the updated ``res``. The default is to append
               # results to a python list. This approach is sufficient for
               # time-series data.
               res.append(results_single_frame)
               # This is not suitable for every analysis. To add results over
               # multiple frames this function can be overwritten. The default
               # value for ``res`` is an empty list. Here we change the type to
               # the return type of `self._single_frame`. Afterwards we can
               # safely use addition to accumulate the results.
               if res == []:
                   res = result_single_frame
               else:
                   res += result_single_frame
               # If you overwrite this function *always* return the updated
               # ``res`` at the end.
               return res

    Afterwards the new analysis can be run like this.

    .. code-block:: python

       na = NewAnalysis(u.select_atoms('name CA'), 35).run()
       print(na.result)

    """

    def __init__(self, universe):
        """Parameters
        ----------
        universe : :class:`~MDAnalysis.core.groups.Universe`
            a :class:`MDAnalysis.core.groups.Universe` (the
            `atomgroups` must belong to this Universe)

        Attributes
        ----------
        _results : list
            The raw data from each process are stored as a list of
            lists, with each sublist containing the return values from
            :meth:`pmda.parallel.ParallelAnalysisBase._single_frame`.

        """
        self._universe = universe
        self._trajectory = universe.trajectory
        #  _dsk keeps the dask graph
        #  (which is a dict of tuples of functions)
        self._dsk = {}
        #  _keys keeps the desired results
        #  in a nested list that represent the outputs of the graph
        self._keys = []
        self._job_prepared = False

    @contextmanager
    def readonly_attributes(self):
        """Set the attributes of this class to be read only

        Useful to avoid the class being modified when passing it around.

        To be used as a context manager::

          with analysis.readonly_attributes():
              some_function(analysis)

        """
        self._attr_lock = True
        yield
        self._attr_lock = False

    def __setattr__(self, key, val):
        # guards to stop people assigning to self when they shouldn't
        # if locked, the only attribute you can modify is _attr_lock
        # if self._attr_lock isn't set, default to unlocked

        # keys that can be changed
        if key in ['_ts', 'prepare_dask_total'] or \
           key == '_attr_lock' or \
           not getattr(self, '_attr_lock', False):
            super().__setattr__(key, val)
        else:
            # raise HalError("I'm sorry Dave, I'm afraid I can't do that")
            raise AttributeError("Can't set '{}' at this time".format(key))

    def _conclude(self):
        """Finalise the results you've gathered.

        Called at the end of the run() method to finish everything up.

        In general this method should unpack :attr:`self._results` to
        sensible variables.

        """
        pass  # pylint: disable=unnecessary-pass

    def _prepare(self):
        """additional preparation to run"""
        pass  # pylint: disable=unnecessary-pass

    def _single_frame(self):
        """Perform computation on a single trajectory frame.

        Must return computed values as a list. You can only **read**
        from member variables stored in ``self``. Changing them during
        a run will result in undefined behavior. `self._ts` and any of the
        atomgroups can be changed (but changes will be overwritten
        when the next time step is read).

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

    def prepare(self,
                start=None,
                stop=None,
                step=None,
                n_jobs=1,
                n_blocks=None):
        """Prepare the jobs

        Parameters
        ----------
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        n_blocks : int, optional
            number of blocks to divide trajectory into. If ``None`` set equal
            to n_jobs or number of available workers in scheduler.
        """
        start, stop, step = self._universe.trajectory.check_slice_indices(start,
                                                                     stop, step)
        n_frames = len(range(start, stop, step))

        self.start, self.stop, self.step = start, stop, step

        self.n_frames = n_frames

        # record prepare time
        with timeit() as prepare_time:
            self._prepare()
        self.time_prepare = prepare_time.elapsed

        #  get global scheduler
        scheduler = dask.config.get('scheduler', None)
        if scheduler == 'processes':
            self.__dask_scheduler__ = dask.multiprocessing.get
        elif scheduler == 'synchronous':
            self.__dask_scheduler__ = dask.get
        elif isinstance(scheduler, dask.distributed.Client):
            self.__dask_scheduler__ = dask.distributed.Client.get
        elif scheduler is not None:
            raise ValueError(f"PMDA doesn't support other"
                             f"schedulers: {scheduler}")

        if n_jobs == -1:
            n_jobs = cpu_count()

        if scheduler is None and n_jobs == 1:
            #  synchronous
            self.__dask_scheduler__ = dask.get

        #  setting n_blocks
        if n_blocks is None:
            if self.__dask_scheduler__ == dask.multiprocessing.get:
                n_blocks = n_jobs
            elif self.__dask_scheduler__ == dask.distributed.Client.get:
                n_blocks = len(dask.distributed.worker.get_client().ncores())
            else:
                n_blocks = 1
                warnings.warn(
                    "Couldn't guess ideal number of blocks from scheduler. "
                    "Setting n_blocks=1. "
                    "Please provide `n_blocks` in call to method.")

        if n_frames == 0:
            warnings.warn("analyses no frames: check start/stop/step")
        if n_frames < n_blocks:
            warnings.warn("uses more blocks than frames: "
                          "will decrease n_blocks")

        slices = make_balanced_slices(n_frames, n_blocks,
                                      start=start, stop=stop, step=step)

        self._blocks = []
        with timeit() as prepare_dask:
            for bslice in slices:
                self._frame_index = bslice

                self._keys.append(self._append_job_to_dsk(self._map_chunk,
                                                          bslice))

                # save the frame numbers for each block
                self._blocks.append(range(bslice.start,
                                          bslice.stop,
                                          bslice.step))

        self.time_prepare_dask = prepare_dask.elapsed
        self._job_prepared = True
        self.wait_start = time.time()
        return self

    def run(self,
            start=None,
            stop=None,
            step=None,
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
        n_jobs : int, optional
            number of jobs to start, if `-1` use number of logical cpu cores.
            This argument will be ignored when the distributed scheduler is
            used
        n_blocks : int, optional
            number of blocks to divide trajectory into. If ``None`` set equal
            to n_jobs or number of available workers in scheduler.
        """
        with timeit() as total:
            if not self._job_prepared:
                self.prepare(start, stop, step, n_jobs, n_blocks)

            if n_jobs == -1:
                n_jobs = cpu_count()
            scheduler_kwargs = {'num_workers': n_jobs}

            _ = self.compute(**scheduler_kwargs)

            #  empty the dask jobs
            self._dsk = {}
            self._keys = []
        self.timing._total = total.elapsed
        return self

    def _map_chunk(self, bslice):
        """function to setup chunk dask graph"""
        # wait_end needs to be first line for accurate timing
        wait_end = time.time()
        res = []
        times_io = []
        times_compute = []
        # NOTE: bslice.stop cannot be None! Always make sure
        #       that it comes from  _trajectory.check_slice_indices()!
        for i in range(bslice.start, bslice.stop, bslice.step):
            self._frame_index = i
            # record io time per frame
            with timeit() as b_io:
                # explicit instead of 'for ts in u.trajectory[bslice]'
                # so that we can get accurate timing.
                self._ts = self._trajectory[i]
            # record compute time per frame
            with timeit() as b_compute:
                res = self._reduce(res, self._single_frame())
            times_io.append(b_io.elapsed)
            times_compute.append(b_compute.elapsed)

        # calculate io and compute time per block
        return np.asarray(res), np.asarray(times_io), np.asarray(
            times_compute), wait_end, np.sum(
            times_io), np.sum(times_compute)

    @staticmethod
    def _reduce(res, result_single_frame):
        """ 'append' action for a time series"""
        res.append(result_single_frame)
        return res

    #  dask-related functions

    def __dask_graph__(self):
        return self._dsk

    def __dask_keys__(self):
        return self._keys

    #  it uses multiprocessing scheduler in default
    __dask_scheduler__ = staticmethod(dask.multiprocessing.get)

    def __dask_postcompute__(self):
        return self._post_reduce, ()

    def _post_reduce(self, res):
        # hack to handle n_frames == 0 in this framework
        if len(res) == 0:
            # everything else wants list of block tuples
            res = [([], [], [], self.wait_start, 0, 0)]
        # record conclude time
        with timeit() as conclude:
            self._results = np.asarray([el[0] for el in res])
            # save the frame numbers for all blocks
            self._conclude()
        # put all time information into the timing object
        self.timing = Timing(
            np.hstack([el[1] for el in res]),
            np.hstack([el[2] for el in res]), 0,
            self.time_prepare,
            self.time_prepare_dask,
            conclude.elapsed,
            # waiting time = wait_end - wait_start
            np.array([el[3] - self.wait_start for el in res]),
            np.array([el[4] for el in res]),
            np.array([el[5] for el in res]))

        #  To make sure the trajectory is reset to initial state,
        #  if we are not running the analysis through the whole trajectory.
        #  With this,  we get the same result (state of the trajectory) from
        #  ParallelAnalysisBase and MDAnalysis.AnalaysisBase.
        self._trajectory.rewind()
        self._dsk = {}
        self._keys = []
        self._job_prepared = False

    def __dask_postpersist__(self):
        #  we don't need persist implementation.
        raise NotImplementedError

    def __dask_tokenize__(self):
        return tuple(self._keys)

    def _append_job_to_dsk(self, func, *args, pure=False, **kwargs):
        #  If pure is True, a consistent hash function is tried on the input.
        #  If False (default), then a unique identifier is always used.

        #  When True, args, current_frame_index, func will all be
        #  used for tokenization. This is much slower and requires the class
        #  to be pickled during this process.
        #  (any reason to use tokenize instead of uuid?

        if not pure:
            name = "%s-%s" % (funcname(func), str(uuid.uuid4()))
        else:
            name = "%s-%s" % (funcname(func), tokenize(func,
                                                       args,
                                                       kwargs,
                                                       self._frame_index))
        self._dsk[name] = (func, *args)
        #  return name to be added to self._keys
        #  which means this function itself does not modify the _keys
        return name
