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

import MDAnalysis as mda

from .parallel import ParallelAnalysisBase


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

    References
    ----------
    .. [Welford1962] B. P. Welford (1962). "Note on a Method for
       Calculating Corrected Sums of Squares and Products." Technometrics
       4(3):419-420.

    See Also
    --------
    MDAnalysis.analysis.density.density_from_Universe


    .. versionadded:: 0.3.0

    """
    def __init__(self, atomgroup):
        self._ag = atomgroup
        super(RMSF, self).__init__(atomgroup.universe, self._ag)

    def _prepare(self):
        self.sumsquares = np.zeros((self.atomgroup.n_atoms, 3))
        self.mean = self.sumsquares.copy()

    def _single_frame(self):
        k = self._frame_index
        self.sumsquares += (k / (k+1.0)) * (self.atomgroup.positions - self.mean) ** 2
        self.mean = (k * self.mean + self.atomgroup.positions) / (k + 1)

    def _conclude(self):
        k = self._frame_index
        self.rmsf = np.sqrt(self.sumsquares.sum(axis=1) / (k + 1))
        
        if not (self.rmsf >= 0).all():
            raise ValueError("Some RMSF values negative; overflow " +
                             "or underflow occurred")

    @staticmethod
    def _reduce(res, result_single_frame):
        res.append(results_single_frame)
        if res == []:
            res = result_single_frame
        else:
            res += result_single_frame
        return res

    @staticmethod
    def pair_wise_rmsf(mu1, mu2, n1, n2, T, M1, M2):
        """
        Calculates the total RMSF pair-wise. Takes in two separate blocks with
        after the RMSF calculation has been concluded and combines their results
        into a single, total RMSF for the combined trajectory slices.

        Parameters
        ----------
        mu1 : (N x 3) NumPy array
            Array of mean positions for each atom in the given atom selection
            and trajectory slice for block 1
        mu2 : (N x 3) NumPy array
            Array of mean positions for each atom in the given atom selection
            and trajectory slice for block 2
        n1 : int
            Number of time steps in trajectory slice 1
        n2 : int
            Number of time steps in trajectory slice 2
        T : int
            Total number of time steps for trajectory
        M1 : (N x 3) NumPy array
            Array of sum of squares for each atom in the given atom selection
            and trajectory slice for block 1
        M2 : (N x 3) NumPy array
            Array of sum of squares for each atom in the given atom selection
            and trajectory slice for block 2
        """
        d_squared = (mu2 - mu1)**2
        return np.sqrt((M1 + M2 + (n1 * n2 * d_squared/T)).sum(axis=1)/T)
