# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
"""
Parallel Analysis building blocks --- :mod:`MDAnalysis.analysis.base`
=====================================================================

A collection of useful building blocks for creating Analysis
classes.

"""
from __future__ import absolute_import
import six
from six.moves import range, zip
import inspect
import logging
import warnings

import numpy as np
from .. import coordinates, Universe
from ..core.groups import AtomGroup
from ..lib.log import ProgressMeter, _set_verbose
from joblib import cpu_count

from dask.delayed import delayed
from dask.distributed import Client
import dask
from dask import multiprocessing
from dask.multiprocessing import get

logger = logging.getLogger(__name__)


class ParallelAnalysisBase(object):
    """Base class for defining parallel multi frame analysis

    The class it is designed as a template for creating multiframe analyses.
    This class will automatically take care of setting up the trajectory
    reader for iterating, and it offers to show a progress meter.

    To define a new Analysis, `AnalysisBase` needs to be subclassed
    `_single_frame` must be defined. It is also possible to define
    `_prepare` and `_conclude` for pre and post processing. See the example
    below.

    .. code-block:: python

       class NewAnalysis(AnalysisBase):
           def __init__(self, atomgroup, parameter, **kwargs):
               super(NewAnalysis, self).__init__(atomgroup.universe.trajectory,
                                                 **kwargs)
               self._parameter = parameter
               self._ag = atomgroup

           def _prepare(self):
               # OPTIONAL
               # Called before iteration on the trajectory has begun.
               # Data structures can be set up at this time
               self.result = []

           def _single_frame(self):
               # REQUIRED
               # Called after the trajectory is moved onto each new frame.
               # store result of `some_function` for a single frame
               self.result.append(some_function(self._ag, self._parameter))

           def _conclude(self):
               # OPTIONAL
               # Called once iteration on the trajectory is finished.
               # Apply normalisation and averaging to results here.
               self.result = np.asarray(self.result) / np.sum(self.result)

    Afterwards the new analysis can be run like this.

    .. code-block:: python

       na = NewAnalysis(u.select_atoms('name CA'), 35).run()
       print(na.result)

    """

    def __init__(self,
                 trajectory,
                 ag,
                 func,
                 start=None,
                 stop=None,
                 step=None,
                 *args,
                 **kwargs):
        """
        Parameters
        ----------
        trajectory : mda.Reader
            A trajectory Reader
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        verbose : bool, optional
            Turn on verbosity
        """
        self._trajectory = trajectory
        self._ag = ag
        start, stop, step = trajectory.check_slice_indices(start, stop, step)
        self.start = start
        self.stop = stop
        self.step = step
        self.n_frames = len(range(start, stop, step))
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def _conclude(self):
        """Finalise the results you've gathered.

        Called at the end of the run() method to finish everything up.
        """
        self.results = np.hstack(self.results)

    def run(self, n_jobs=1):
        """Perform the calculation"""
        if n_jobs == -1:
            n_jobs = cpu_count()
        n_blocks = n_jobs
        bsize = int(np.ceil(self.n_frames / float(n_blocks)))
        blocks = []
        top = self._ag.universe.filename
        traj = self._ag.universe.trajectory.filename
        indices = self._ag.indices

        for b in range(n_blocks):
            start = b * bsize + self.start
            stop = (b + 1) * bsize * self.step
            task = delayed(
                self.para_helper,
                pure=False)(start, stop, self.step, indices, top, traj,
                            *self.args, **self.kwargs)
            blocks.append(task)
        blocks = delayed(blocks)
        self.results = blocks.compute()

        self._conclude()
        return self

    def para_helper(self, start, stop, step, indices, top, traj, *args,
                    **kwargs):
        u = Universe(top, traj)
        ag = u.atoms[indices]

        res = []
        for i, ts in enumerate(u.trajectory[start:stop:step]):
            res.append(self.func(ag, *args, **kwargs))

        return np.asarray(res)
