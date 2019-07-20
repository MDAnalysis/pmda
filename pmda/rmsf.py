# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2019 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version

"""
Calculating Root-Mean-Square Fluctuations (RMSF) --- :mod:`pmda.rmsf`
=====================================================================

This module contains parallel versions of analysis tasks in
:mod:`MDAnalysis.analysis.rms`.

.. autoclass:: RMSF
   :members:
   :undoc-members:
   :inherited-members:

See Also
--------
MDAnalysis.analysis.rms.RMSF

"""

from __future__ import absolute_import

import numpy as np

from .parallel import ParallelAnalysisBase

from .util import pair_wise_rmsf


class RMSF(ParallelAnalysisBase):
    r"""Parallel RMSF Analysis.

    Calculates RMSF of given atoms across a trajectory.

    Parameters
    ----------
    atomgroup : AtomGroup
        Atoms for which RMSF is calculated

    Raises
    ------
    ValueError
    raised if negative values are calculated, which indicates that a
    numerical overflow or underflow occured

    See Also
    --------
    MDAnalysis.analysis.rms.RMSF

    Notes
    -----
    No RMSD-superposition is performed; it is assumed that the user is
    providing a trajectory where the protein of interest has been structurally
    aligned to a reference structure (see the Examples section below). The
    protein also has be whole because periodic boundaries are not taken into
    account.
    Run the analysis with :meth:`RMSF.run`, which stores the results
    in the array :attr:`RMSF.rmsf`.

    The root mean square fluctuation of an atom :math:`i` is computed as the
    time average
    .. math::
    \rho_i = \sqrt{\left\langle (\mathbf{x}_i - \langle\mathbf{x}_i\rangle)^2 \right\rangle}
    No mass weighting is performed.
    This method implements an algorithm for computing sums of squares while
    avoiding overflows and underflows [Welford1962]_.

    References
    ----------
    .. [Welford1962] B. P. Welford (1962). "Note on a Method for
    Calculating Corrected Sums of Squares and Products." Technometrics
    4(3):419-420.

    Examples
    --------
    In this example we calculate the residue RMSF fluctuations by analyzing
    the :math:`\text{C}_\alpha` atoms. First we need to fit the trajectory
    to the average structure as a reference. That requires calculating the
    average structure first. Because we need to analyze and manipulate the
    same trajectory multiple times, we are going to load it into memory
    using the :mod:`~MDAnalysis.coordinates.MemoryReader`. (If your
    trajectory does not fit into memory, you will need to :ref:`write out
    intermediate trajectories <writing-trajectories>` to disk or
    :ref:`generate an in-memory universe
    <creating-in-memory-trajectory-label>` that only contains, say, the
    protein)::

       import MDAnalysis as mda
       from MDAnalysis.analysis import align
       from MDAnalysis.tests.datafiles import TPR, XTC
       u = mda.Universe(TPR, XTC, in_memory=True)
       protein = u.select_atoms("protein")
       # 1) need a step to center and make whole: this trajectory
       #    contains the protein being split across periodic boundaries
       #
       # TODO
       # 2) fit to the initial frame to get a better average structure
       #    (the trajectory is changed in memory)
       prealigner = align.AlignTraj(u, select="protein and name CA",
                                    in_memory=True).run()
       # 3) ref = average structure
       ref_coordinates = u.trajectory.timeseries(asel=protein).mean(axis=1)
       # make a reference structure (need to reshape into a 1-frame "trajectory")
       ref = mda.Merge(protein).load_new(ref_coordinates[:, None, :],
                                         order="afc")

    We created a new universe ``reference`` that contains a single frame
    with the averaged coordinates of the protein.  Now we need to fit the
    whole trajectory to the reference by minimizing the RMSD. We use
    :class:`MDAnalysis.analysis.align.AlignTraj`::

       aligner = align.AlignTraj(u, reference, select="protein and name CA",
                                 in_memory=True).run()

    The trajectory is now fitted to the reference (the RMSD is stored as
    `aligner.rmsd` for further inspection). Now we can calculate the RMSF::

       from MDAnalysis.analysis.rms import RMSF
       calphas = protein.select_atoms("name CA")
       rmsfer = RMSF(calphas, verbose=True).run()

    and plot::

       import matplotlib.pyplot as plt
       plt.plot(calphas.resnums, rmsfer.rmsf)


    .. versionadded:: 0.3.0

    """
    def __init__(self, atomgroup):
        u = atomgroup.universe
        super(RMSF, self).__init__(u, (atomgroup, ))
        self._atomgroup = atomgroup
        self._top = u.filename
        self._traj = u.trajectory.filename

    def _prepare(self):
        self.sumsquares = np.zeros((self._atomgroup.n_atoms, 3))
        self.mean = self.sumsquares.copy()

    def _single_frame(self, ts, agroups):
        k = ts.frame
        sumsquares = self.sumsquares
        mean = self.mean
        agroups = agroups[0]
        return k, agroups, sumsquares, mean

    def _conclude(self):
        """
        self._results : Array
            (n_blocks x ts x 2 x N x 3) array
        """
        k = len(self._trajectory)
        self.sumsquares = self._results[0, 0]
        self.mean = self._results[0, 1]
        self.rmsf = np.sqrt(self.sumsquares.sum(axis=1) / k)
        if not (self.rmsf >= 0).all():
            raise ValueError("Some RMSF values negative; overflow " +
                             "or underflow occurred")

    @staticmethod
    def _reduce(res, frame_result):
        """
        'append' action for a time series
        """
        n = frame_result[0]
        positions = frame_result[1].positions
        # for initial time step
        if n == 0:
            # assign inital mean and sum of squares zero-arrays to res
            res.append(frame_result[2])
            res.append(frame_result[3])
        else:
            # retrieve mean from previous time step
            mean = res[1]
            # update mean and sum of squares
            res[0] += (n / (n + 1)) * (positions - mean) ** 2
            res[1] = (n * mean + positions) / (n + 1)
        return res
