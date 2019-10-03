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

    .. attribute:: rmsf

        Results are stored in this N-length :class:`numpy.ndarray` array,
        giving RMSFs for each of the given atoms.

See Also
--------
MDAnalysis.analysis.rms.RMSF

"""

from __future__ import absolute_import, division

import numpy as np

from pmda.parallel import ParallelAnalysisBase

from pmda.util import fold_second_order_moments


class RMSF(ParallelAnalysisBase):
    r"""Parallel RMSF analysis.

    Calculates RMSF of given atoms across a trajectory.

    Attributes
    ----------
    rmsf : array
         N-length :class:`numpy.ndarray` array of RMSF values, where `N` is
         the number of atoms in the atomgroup of interest. Returned values
         have units of ångströms.

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
    time average:

    .. math::

        \sigma_{i} = \sqrt{\left\langle (\mathbf{x}_{i} -
                           \langle\mathbf{x}_{i}\rangle)^2
                           \right\rangle}

    No mass weighting is performed.
    This method implements an algorithm for computing sums of squares while
    avoiding overflows and underflows [Welford1962]_, as well as an algorithm
    for combining the sum of squares and means of separate partitions of a
    given trajectory to calculate the RMSF of the entire trajectory
    [CGL1979]_.

    For more details about RMSF calculations, refer to [Awtrey2019]_.

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

        # TODO: Need to center and make whole (this test trajectory
        # contains the protein being split across periodic boundaries
        # and the results will be WRONG!)

        # Fit to the initial frame to get a better average structure
        # (the trajectory is changed in memory)
        prealigner = align.AlignTraj(u, u, select="protein and name CA",
                                     in_memory=True).run()
        # ref = average structure
        ref_coordinates = u.trajectory.timeseries(asel=protein).mean(axis=1)
        # Make a reference structure (need to reshape into a
        # 1-frame "trajectory").
        ref = mda.Merge(protein).load_new(ref_coordinates[:, None, :],
                                          order="afc")

    We created a new universe ``reference`` that contains a single frame
    with the averaged coordinates of the protein.  Now we need to fit the
    whole trajectory to the reference by minimizing the RMSD. We use
    :class:`MDAnalysis.analysis.align.AlignTraj`::

        aligner = align.AlignTraj(u, ref, select="protein and name CA",
                                  in_memory=True).run()
        # need to write the trajectory to disk for PMDA 0.3.0 (see issue #15)
        with mda.Writer("rmsfit.xtc", n_atoms=u.atoms.n_atoms) as W:
            for ts in u.trajectory:
                W.write(u.atoms)

    (For use with PMDA we cannot currently use a in-memory trajectory
    (see `Issue #15`_) so we must write out the RMS-superimposed
    trajectory to the file "rmsfit.xtc".)

    The trajectory is now fitted to the reference (the RMSD is stored as
    `aligner.rmsd` for further inspection). Now we can calculate the RMSF::

        from pmda.rms import RMSF

        u = mda.Universe(TPR, "rmsfit.xtc")
        calphas = protein.select_atoms("protein and name CA")

        rmsfer = RMSF(calphas).run()

    and plot::

        import matplotlib.pyplot as plt
        plt.plot(calphas.resnums, rmsfer.rmsf)


    .. versionadded:: 0.3.0


    .. _`Issue #15`: https://github.com/MDAnalysis/pmda/issues/15

    """

    def __init__(self, atomgroup):
        u = atomgroup.universe
        super(RMSF, self).__init__(u, (atomgroup, ))
        self._atomgroup = atomgroup
        self._top = u.filename
        self._traj = u.trajectory.filename

    def _single_frame(self, ts, atomgroups):
        # mean and sum of squares calculations done in _reduce()
        return atomgroups[0]

    def _conclude(self):
        """
        self._results : Array
            (n_blocks x 2 x N x 3) array
        """
        n_blocks = len(self._results)
        # serial case
        if n_blocks == 1:
            # get length of trajectory slice
            self.mean = self._results[0, 0]
            self.sumsquares = self._results[0, 1]
            self.rmsf = np.sqrt(self.sumsquares.sum(axis=1) / self.n_frames)
        # parallel case
        else:
            mean = self._results[:, 0]
            sos = self._results[:, 1]
            # create list of (timesteps, mean, sumsq tuples for each block
            vals = []
            for i in range(n_blocks):
                vals.append((len(self._blocks[i]), mean[i], sos[i]))
            # combine block results using fold method
            results = fold_second_order_moments(vals)
            self.mean = results[1]
            self.sumsquares = results[2]
            self.rmsf = np.sqrt(self.sumsquares.sum(axis=1) / self.n_frames)
            self._negative_rmsf(self.rmsf)

    @staticmethod
    def _reduce(res, result_single_frame):
        """
        'sum' action for time series
        """
        atoms = result_single_frame
        positions = atoms.positions.astype(np.float64)
        # initial time step case
        if isinstance(res, list) and len(res) == 0:
            # initial mean position = initial position
            mean = positions
            # create new zero-array for sum of squares to prevent blocks from
            # using data from previous blocks
            sumsq = np.zeros((atoms.n_atoms, 3))
            # set initial time step for each block to zero
            k = 0
            # assign initial (sum of squares and mean) zero-arrays to res
            res = [mean, sumsq, k]
        else:
            # update time step
            k = res[2] + 1
            # update sum of squares
            res[1] += (k / (k + 1)) * (positions - res[0]) ** 2
            # update mean
            res[0] = (k * res[0] + positions) / (k + 1)
            # update time step in res
            res[2] = k
        return res

    @staticmethod
    def _negative_rmsf(rmsf):
        if not (rmsf >= 0).all():
            raise ValueError("Some RMSF values negative; overflow " +
                             "or underflow occurred")
