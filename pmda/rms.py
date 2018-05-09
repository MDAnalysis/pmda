# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
"""
RMSD analysis tools --- :mod:`pmda.rms`
=======================================

This module contains parallel versions of analysis tasks in
:mod:`MDAnalysis.analysis.rms`.

.. autoclass:: RMSD
   :members:
   :undoc-members:
   :inherited-members:

"""
from __future__ import absolute_import

from MDAnalysis.analysis import rms
import numpy as np

from .parallel import ParallelAnalysisBase


class RMSD(ParallelAnalysisBase):
    """Parallel RMSD analysis.

    Optimally superimpose the coordinates in the
    :class:`~MDAnalysis.core.groups.AtomGroup` `mobile` onto `ref` for
    each frame in the trajectory of `mobile` and calculate the time
    series of the RMSD. The single frame calculation is performed with
    :func:`MDAnalysis.analysis.rms.rmsd` (with ``superposition=True``
    by default).

    Attributes
    ----------
    rmsd : array
         Result as a Nx3 array where each row contains
         `[frame, time (ps), RMSD (Å)]`.


    Parameters
    ----------
    mobile : AtomGroup
         atoms that are optimally superimposed on `ref` before
         the RMSD is calculated for all atoms. The coordinates
         of `mobile` change with each frame in the trajectory.
    ref : AtomGroup
         fixed reference coordinates
    superposition : bool, optional
         ``True`` perform a RMSD-superposition, ``False`` only
         calculates the RMSD. The default is ``True``.


    Note
    ----
    At the moment, this class has far fewer features than the serial
    version :class:`MDAnalysis.analysis.rms.RMSD`.

    """
    def __init__(self, mobile, ref, superposition=True):
        universe = mobile.universe
        super(RMSD, self).__init__(universe, (mobile, ))
        self._ref_pos = ref.positions.copy()
        self.superposition = superposition

    def _prepare(self):
        self.rmsd = None

    def _conclude(self):
        self.rmsd = np.vstack(self._results)

    def _single_frame(self, ts, atomgroups):
        return (ts.frame, ts.time,
                rms.rmsd(atomgroups[0].positions, self._ref_pos,
                         superposition=self.superposition))
