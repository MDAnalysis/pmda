# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2019 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version

"""
Calculating Root-Mean-Square Deviations (RMSD) --- :mod:`pmda.rms`
=====================================================================

This module contains parallel versions of analysis tasks in
:mod:`MDAnalysis.analysis.rms.RMSD`.

.. autoclass:: RMSD
    :members:

    .. attribute:: rmsd

        Contains the time series of the RMSD as a `Tx3` :class:`numpy.ndarray`
        array with content ``[[frame, time (ps), RMSD (Å)], [...], ...]``,
        where `T` is the number of time steps selected in the :meth:`run`
        method.

See Also
--------
MDAnalysis.analysis.rms.RMSD

"""
from __future__ import absolute_import

from MDAnalysis.analysis import rms

import numpy as np

from pmda.parallel import ParallelAnalysisBase


class RMSD(ParallelAnalysisBase):
    r"""Parallel RMSD analysis.

    Optimally superimpose the coordinates in the
    :class:`~MDAnalysis.core.groups.AtomGroup` `mobile` onto `ref` for
    each frame in the trajectory of `mobile`, and calculate the time
    series of the RMSD. The single frame calculation is performed with
    :func:`MDAnalysis.analysis.rms.rmsd` (with ``superposition=True``
    by default).

    Attributes
    ----------
    rmsd : array
         `Tx3` array where each row contains
         `[frame, time (ps), RMSD (Å)]`, and `T` is the number of time steps
         selected in the :meth:`run` method.

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

    See Also
    --------
    MDAnalysis.analysis.rms.RMSD

    Notes
    -----
    If you use trajectory data from simulations performed under *periodic
    boundary conditions* then you **must make your molecules whole** before
    performing RMSD calculations so that the centers of mass of the mobile and
    reference structure are properly superimposed.

    Run the analysis with :meth:`RMSD.run`, which stores the results
    in the array :attr:`RMSD.rmsd`.

    The root mean square deviation :math:`\rho(t)` of a group of :math:`N`
    atoms relative to a reference structure as a function of time is
    calculated as:

    .. math::

        \rho(t) = \sqrt{\frac{1}{N} \sum_{i=1}^N \left(\mathbf{x}_i(t)
                        - \mathbf{x}_i^{\text{ref}}\right)^2}

    The selected coordinates from `atomgroup` are optimally superimposed
    (translation and rotation) on the `reference` coordinates at each time step
    as to minimize the RMSD.

    At the moment, this class has far fewer features than its serial
    counterpart, :class:`MDAnalysis.analysis.rms.RMSD`.

    Examples
    --------
    In this example we will globally fit a protein to a reference
    structure. The example is a DIMS trajectory of adenylate kinase, which
    samples a large closed-to-open transition.

    The trajectory is included in the MDAnalysis test data files. The data in
    :attr:`RMSD.rmsd` is plotted with :func:`matplotlib.pyplot.plot`::

        import MDAnalysis
        from MDAnalysis.tests.datafiles import PSF, DCD, CRD
        mobile = MDAnalysis.Universe(PSF,DCD).atoms
        # reference closed AdK (1AKE) (with the default ref_frame=0)
        ref = MDAnalysis.Universe(PSF,DCD).atoms

        from pmda.rms import RMSD

        R = RMSD(mobile, ref)
        R.run()

        import matplotlib.pyplot as plt
        rmsd = R.rmsd.T[2]  # transpose makes it easier for plotting
        time = rmsd[0]
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.plot(time, rmsd,  label="all")
        ax.legend(loc="best")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel(r"RMSD ($\\AA$)")
        fig.savefig("rmsd_all_CORE_LID_NMP_ref1AKE.pdf")

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
