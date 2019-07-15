import numpy as np
import MDAnalysis as mda
import time
from MDAnalysis.lib.util import fixedwidth_bins
from MDAnalysis.analysis.density import Density
from .parallel import ParallelAnalysisBase


class DensityAnalysis(ParallelAnalysisBase):
    """Parallel density analysis.

    Parameters
    ----------
    atomgroup : AtomGroup
            Group of atoms (such as all the water oxygen atoms) being analyzed.
            If this is an updating AtomGroup then you need to set
            'atomselection' and also 'updating=True'.
    atomselection : str (optional)
            selection string (MDAnalysis syntax) for the species to be analyzed
            ["name OH2"]
    delta : float (optional)
            bin size for the density grid in Angstroem (same in x,y,z) [1.0]
            Slice the trajectory as "trajectory[start:stop:step]"; default
            is to read the whole trajectory.
    metadata : dict. (optional)
            `dict` of additional data to be saved with the object; the meta
            data are passed through as they are.
    padding : float (optional)
            increase histogram dimensions by padding (on top of initial box
            size) in Angstroem. Padding is ignored when setting a user defined
            grid. [2.0]
    updating : bool (optional)
            Should the selection of atoms be updated for every step? ["False"]
            - "True": atom selection is updated for each frame, can be slow
            - "False": atoms are only selected at the beginning
    parameters : dict (optional)
            `dict` with some special parameters for :class:`Density` (see docs)
    gridcenter : numpy ndarray, float32 (optional)
            3 element numpy array detailing the x, y and z coordinates of the
            center of a user defined grid box in Angstroem ["None"]
    xdim : float (optional)
            User defined x dimension box edge in ångström; ignored if
            gridcenter is "None"
    ydim : float (optional)
            User defined y dimension box edge in ångström; ignored if
            gridcenter is "None"
    zdim : float (optional)
            User defined z dimension box edge in ångström; ignored if
            gridcenter is "None"
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
        self._n_frames = u.trajectory.n_frames
        if updating and atomselection is None:
            raise ValueError("updating=True requires a atomselection string")
        elif not updating and atomselection is not None:
            raise ValueError("With updating=False, the atomselection='{}' is not
            used and should be None".format(atomselection))

    def _prepare(self):
        coord = self.current_coordinates(self._atomgroup, self._atomselection,
                                         self._updating)
        box, angles = (self._trajectory.ts.dimensions[:3],
                       self._trajectory.ts.dimensions[3:])
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
        h, edges = np.histogramdd(coord, bins=self._bins, range=self._arange,
                                  normed=False)
        return h

    def _conclude(self):
        self._grid = self._results[:].sum(axis=0)
        self._grid /= float(self._n_frames)
        metadata = self._metadata if self._metadata is not None else {}
        metadata['psf'] = self._atomgroup.universe.filename
        metadata['dcd'] = self._trajectory.filename
        metadata['atomselection'] = self._atomselection
        metadata['n_frames'] = self._n_frames
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
        ag = (atomgroup if not updating else
        atomgroup.select_atoms(atomselection))
        return ag.positions
