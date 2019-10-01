# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2019 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
"""
Generating Densities from Trajectories --- :mod:`pmda.density`
===============================================================

This module contains parallel versions of analysis tasks in
:mod:`MDAnalysis.analysis.density`.

.. autoclass:: DensityAnalysis
   :members:
   :inherited-members:

See Also
--------
MDAnalysis.analysis.density.density_from_Universe

"""

from __future__ import absolute_import

import numpy as np

import MDAnalysis as mda

from MDAnalysis.lib.util import fixedwidth_bins

from MDAnalysis.analysis.density import Density

from MDAnalysis.analysis.density import _set_user_grid

from .parallel import ParallelAnalysisBase


class DensityAnalysis(ParallelAnalysisBase):
    r"""Parallel density analysis.

    The trajectory is read, frame by frame, and the atoms selected with
    `atomselection` are histogrammed on a grid with spacing `delta`.

    Parameters
    ----------
    atomgroup : AtomGroup
            Group of atoms (such as all the water oxygen atoms) being analyzed.
            If this is an updating AtomGroup then you need to set
            'atomselection' and also 'updating=True'.
    atomselection : str (optional)
            Selection string (MDAnalysis syntax) for the species to be analyzed
            ["name OH2"]
    delta : float (optional)
            Bin size for the density grid in ångström (same in x,y,z).
            Slice the trajectory as "trajectory[start:stop:step]"; default
            is to read the whole trajectory.
    metadata : dict (optional)
            `dict` of additional data to be saved with the object; the meta
            data are passed through as they are.
    padding : float (optional)
            Increase histogram dimensions by padding (on top of initial box
            size) in ångström. Padding is ignored when setting a user defined
            grid.
    updating : bool (optional)
            Should the selection of atoms be updated for every step? ["False"]
            - "True": atom selection is updated for each frame, can be slow
            - "False": atoms are only selected at the beginning
    parameters : dict (optional)
            `dict` with some special parameters for :class:`Density` (see docs)
    gridcenter : numpy ndarray, float32 (optional)
            3 element numpy array detailing the x, y and z coordinates of the
            center of a user defined grid box in ångström.
    xdim : float (optional)
            User defined x dimension box edge in ångström; ignored if
            gridcenter is "None".
    ydim : float (optional)
            User defined y dimension box edge in ångström; ignored if
            gridcenter is "None".
    zdim : float (optional)
            User defined z dimension box edge in ångström; ignored if
            gridcenter is "None".

    See Also
    --------
    MDAnalysis.analysis.density.density_from_Universe

    Notes
    -----
    By default, the `atomselection` is static, i.e., atoms are only selected
    once at the beginning. If you want *dynamically changing selections*
    (such as "name OW and around 4.0 (protein and not name H*)", i.e., the
    water oxygen atoms that are within 4 Å of the protein heavy atoms) then
    set ``updating=True``.

    For more details about density calculations, refer to [Awtrey2019]_.

    Examples
    --------
    A common use case is to analyze the solvent density around a protein of
    interest. The density is calculated with :class:`DensityAnalysis` in the
    fixed coordinate system of the simulation unit cell. It is therefore
    necessary to orient and fix the protein with respect to the box coordinate
    system. In practice this means centering and superimposing the protein,
    frame by frame, on a reference structure and translating and rotating all
    other components of the simulation with the protein. In this way, the
    solvent will appear in the reference frame of the protein.

    An input trajectory must

    1. have been centered on the protein of interest;
    2. have all molecules made whole that have been broken across periodic
       boundaries [#pbc]_;
    3. have the solvent molecules remapped so that they are closest to the
       solute (this is important when using triclinic unit cells such as
       a dodecahedron or a truncated octahedron) [#pbc]_;
    4. have a fixed frame of reference; for instance, by superimposing a
       protein on a reference structure so that one can study the solvent
       density around it [#fit]_.

    To generate the density of water molecules around a protein (assuming that
    the trajectory is already appropriately treated for periodic boundary
    artifacts and is suitably superimposed to provide a fixed reference frame)
    [#testraj]_, first  create the :class:`DensityAnalysis` object by
    supplying an AtomGroup, then use  the :meth:`run` method::

        from pmda.density import DensityAnalysis
        U = Universe(TPR, XTC)
        ow = U.select_atoms("name OW")
        D = DensityAnalysis(ow, delta=1.0)
        D.run()
        D.convert_density('TIP4P')

    The positions of all water oxygens are histogrammed on a grid with spacing
    *delta* = 1 Å. Initially the density is measured in inverse cubic
    angstroms. With the :meth:`Density.convert_density` method,
    the units of measurement are changed. In the example we are now measuring
    the density relative to the literature value of the TIP4P water model at
    ambient conditions (see the values in :data:`MDAnalysis.units.water` for
    details). In particular, the density is stored as a NumPy array in
    :attr:`grid`, which can be processed in any manner. Results are available
    through the :attr:`density` attribute, which has the :attr:`grid`
    attribute that contains the histogrammed density data.
    :attr:`DensityAnalysis.density` is a :class:`gridData.core.Grid` object.
    In particular, its contents can be `exported to different formats
    <https://www.mdanalysis.org/GridDataFormats/gridData/formats.html>`_.
    For example, to `write a DX file
    <https://www.mdanalysis.org/GridDataFormats/gridData/basic.html#writing-out-data>`_
    ``water.dx`` that can be read with VMD, PyMOL, or Chimera::

      D.density.export("water.dx", type="double")

    Basic use for creating a water density (just using the water oxygen
    atoms "OW")::

      D = DensityAnalysis(universe.atoms, atomselection='name OW')

    If you are only interested in water within a certain region, e.g., within
    a vicinity around a binding site, you can use a selection that updates
    every step by setting the `updating` keyword argument::

      atomselection = 'name OW and around 5 (resid 156 157 305)'
      D_site = DensityAnalysis(universe.atoms, atomselection=atomselection,
                               updating=True)

    If you are interested in explicitly setting a grid box of a given edge
    size and origin, you can use the gridcenter and x/y/zdim arguments.
    For example to plot the density of waters within 5 Å of a ligand (in this
    case the ligand has been assigned the residue name "LIG") in a cubic grid
    with 20 Å edges which is centered on the center of mass (COM) of the
    ligand::

      # Create a selection based on the ligand
      ligand_selection = universe.select_atoms("resname LIG")

      # Extract the COM of the ligand
      ligand_COM = ligand_selection.center_of_mass()

      # Create a density of waters on a cubic grid centered on the ligand COM
      # In this case, we update the atom selection as shown above.
      D_water = DensityAnalysis(universe.atoms, delta=1.0,
                                atomselection='name OW around 5 resname LIG',
                                updating=True,
                                gridcenter=ligand_COM,
                                xdim=20, ydim=20, zdim=20)

      (It should be noted that the `padding` keyword is not used when a user
      defined grid is assigned).

    .. rubric:: Footnotes

    .. [#pbc] Making molecules whole can be accomplished with the
              :meth:`MDAnalysis.core.groups.AtomGroup.wrap` of
              :attr:`Universe.atoms` (use ``compound="fragments"``).

              When using, for instance, the Gromacs_ command `gmx trjconv`_::

                gmx trjconv -pbc mol -center -ur compact

              one can make the molecules whole ``-pbc whole``, center it on a
              group (``-center``), and also pack all molecules in a compact
              unitcell representation, which can be useful for density
              generation.

    .. [#fit] Superposition can be performed with
              :class:`MDAnalysis.analysis.align.AlignTraj`.

              The Gromacs_ command `gmx trjconv`_::

                gmx trjconv -fit rot+trans

              will also accomplish such a superposition. Note that the fitting
              has to be done in a *separate* step from the treatment of the
              periodic boundaries [#pbc]_.

    .. [#testraj] Note that the trajectory in the example (`XTC`) is *not*
                  properly made whole and fitted to a reference structure;
                  these steps were omitted to clearly show the steps necessary
                  for the actual density calculation.

    .. Links
    .. -----
    .. _OpenDX: http://www.opendx.org/
    .. _VMD:   http://www.ks.uiuc.edu/Research/vmd/
    .. _Chimera: https://www.cgl.ucsf.edu/chimera/
    .. _PyMOL: http://www.pymol.org/
    .. _Gromacs: http://www.gromacs.org
    .. _`gmx trjconv`: http://manual.gromacs.org/programs/gmx-trjconv.html


    .. versionadded:: 0.3.0

    """
    def __init__(self, atomgroup, delta=1.0, atomselection=None,
                 metadata=None, padding=2.0, updating=False,
                 parameters=None, gridcenter=None, xdim=None, ydim=None,
                 zdim=None):
        u = atomgroup.universe
        super(DensityAnalysis, self).__init__(u, (atomgroup, ))
        self._atomgroup = atomgroup
        self._delta = delta
        self._atomselection = atomselection
        self._metadata = metadata
        self._padding = padding
        self._updating = updating
        self._parameters = parameters
        self._gridcenter = gridcenter
        self._xdim = xdim
        self._ydim = ydim
        self._zdim = zdim
        self._trajectory = u.trajectory
        if updating and atomselection is None:
            raise ValueError("updating=True requires a atomselection string")
        elif not updating and atomselection is not None:
            raise ValueError("""With updating=False, the atomselection='{}' is
                        not used and should be None""".format(atomselection))

    def _prepare(self):
        coord = self.current_coordinates(self._atomgroup, self._atomselection,
                                         self._updating)
        if self._gridcenter is not None:
            # Generate a copy of smin/smax from coords to later check if the
            # defined box might be too small for the selection
            smin = np.min(coord, axis=0)
            smax = np.max(coord, axis=0)
            # Overwrite smin/smax with user defined values
            smin, smax = _set_user_grid(self._gridcenter, self._xdim,
                                        self._ydim, self._zdim, smin, smax)
        else:
            # Make the box bigger to avoid as much as possible 'outlier'. This
            # is important if the sites are defined at a high density: in this
            # case the bulk regions don't have to be close to 1 * n0 but can
            # be less. It's much more difficult to deal with outliers.  The
            # ideal solution would use images: implement 'looking across the
            # periodic boundaries' but that gets complicated when the box
            # rotates due to RMS fitting.
            smin = np.min(coord, axis=0) - self._padding
            smax = np.max(coord, axis=0) + self._padding
        BINS = fixedwidth_bins(self._delta, smin, smax)
        arange = np.transpose(np.vstack((BINS['min'], BINS['max'])))
        bins = BINS['Nbins']
        # create empty grid with the right dimensions (and get the edges)
        grid, edges = np.histogramdd(np.zeros((1, 3)), bins=bins,
                                     range=arange, normed=False)
        grid *= 0.0
        self._grid = grid
        self._edges = edges
        self._arange = arange
        self._bins = bins

    def _single_frame(self, ts, atomgroups):
        coord = self.current_coordinates(atomgroups[0], self._atomselection,
                                         self._updating)
        result = np.histogramdd(coord, bins=self._bins, range=self._arange,
                                normed=False)
        return result[0]

    def _conclude(self):
        self._grid = self._results[:].sum(axis=0)
        self._grid /= float(self.n_frames)
        metadata = self._metadata if self._metadata is not None else {}
        metadata['psf'] = self._atomgroup.universe.filename
        metadata['dcd'] = self._trajectory.filename
        metadata['atomselection'] = self._atomselection
        metadata['n_frames'] = self.n_frames
        metadata['totaltime'] = self._atomgroup.universe.trajectory.totaltime
        metadata['dt'] = self._trajectory.dt
        metadata['time_unit'] = mda.core.flags['time_unit']
        parameters = self._parameters if self._parameters is not None else {}
        parameters['isDensity'] = False  # must override
        density = Density(grid=self._grid, edges=self._edges,
                          units={'length': "Angstrom"},
                          parameters=parameters,
                          metadata=metadata)
        density.make_density()
        self.density = density

    @staticmethod
    def _reduce(res, result_single_frame):
        """ 'accumulate' action for a time series"""
        if isinstance(res, list) and len(res) == 0:
            res = result_single_frame
        else:
            res += result_single_frame
        return res

    @staticmethod
    def current_coordinates(atomgroup, atomselection, updating):
        """Retrieves the current coordinates of all atoms in the chosen atom
        selection.
        Note: currently required to allow for updating selections"""
        ag = atomgroup if not updating else atomgroup.select_atoms(
                                                            atomselection)
        return ag.positions
