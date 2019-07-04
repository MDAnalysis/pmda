import numpy as np
import MDAnalysis as mda
import time
from MDAnalysis.lib.util import fixedwidth_bins
from MDAnalysis.analysis.density import Density
from .parallel import ParallelAnalysisBase

class pDensity(ParallelAnalysisBase):
    """Parallel density analysis.

    Parameters
    ----------
    universe : MDAnalysis.Universe
            :class:`MDAnalysis.Universe` object with a trajectory
    atomselection : str (optional)
            selection string (MDAnalysis syntax) for the species to be analyzed
            ["name OH2"]
    delta : float (optional)
            bin size for the density grid in Angstroem (same in x,y,z) [1.0]
    start : int (optional)
    stop : int (optional)
    step : int (optional)
            Slice the trajectory as "trajectory[start:stop:step]"; default
            is to read the whole trajectory.
    metadata : dict. optional
            `dict` of additional data to be saved with the object; the meta data
            are passed through as they are.
    padding : float (optional)
            increase histogram dimensions by padding (on top of initial box size)
            in Angstroem. Padding is ignored when setting a user defined grid. [2.0]
    soluteselection : str (optional)
            MDAnalysis selection for the solute, e.g. "protein" ["None"]
    cutoff : float (optional)
            With `cutoff`, select "<atomsel> NOT WITHIN <cutoff> OF <soluteselection>"
            (Special routines that are faster than the standard "AROUND" selection);
            any value that evaluates to "False" (such as the default 0) disables this
            special selection.
    update_selection : bool (optional)
            Should the selection of atoms be updated for every step? ["False"]
            - "True": atom selection is updated for each frame, can be slow
            - "False": atoms are only selected at the beginning
    verbose : bool (optional)
            Print status update to the screen for every *interval* frame? ["True"]
            - "False": no status updates when a new frame is processed
            - "True": status update every frame (including number of atoms
              processed, which is interesting with "update_selection=True")
    interval : int (optional)
           Show status update every `interval` frame [1]
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

    Returns
    -------
    :class:`Density`
            A :class:`Density` instance with the histogrammed data together
            with associated metadata.
    """"

    def __init__(self, atomgroup, delta=1.0, atomselection="name OH2", start=None,
                stop=None, step=None, metadata=None, padding=2.0, cutoff=0,
                soluteselection=None, use_kdtree=True, update_selection=False,
                verbose=False, interval=1, quiet=None, parameters=None,
                gridcenter=None, xdim=None, ydim=None, zdim=None):
        """
        Parameters
        ----------
        atomgroup : AtomGroup
            atom groups that are iterated over in parallel
        """
        u = atomgroup.universe
        super(pDensity, self).__init__(u, (atomgroup, ))
        self._atomgroup = atomgroup
        self._trajectory = u.trajectory
        self._cutoff = cutoff
        self._atomselection = atomselection
        self._soluteselection = soluteselection
        self._update_selection = update_selection
        self._padding = padding
        self._metadata = metadata
        self._parameters = parameters
        self._n_frames = u.trajectory.n_frames
        coord = self.current_coordinates(atomgroup, cutoff, atomselection, soluteselection, update_selection)
        box, angles = self._trajectory.ts.dimensions[:3], self._trajectory.ts.dimensions[3:]
        # logger.info(
        #     "Selected {0:d} atoms out of {1:d} atoms ({2!s}) from {3:d} total."
        #     "".format(coord.shape[0], len(u.select_atoms(atomselection)),
        #               atomselection, len(u.atoms))
        # )

        # mild warning; typically this is run on RMS-fitted trajectories and
        # so the box information is rather meaningless
        # if tuple(angles) != (90., 90., 90.):
        #     msg = "Non-orthorhombic unit-cell --- make sure that it has been remapped properly!"
        #     warnings.warn(msg)
        #     logger.warning(msg)

        if gridcenter is not None:
            # Generate a copy of smin/smax from coords to later check if the
            # defined box might be too small for the selection
            smin = np.min(coord, axis=0)
            smax = np.max(coord, axis=0)
            # Overwrite smin/smax with user defined values
            smin, smax = _set_user_grid(gridcenter, xdim, ydim, zdim, smin, smax)
        else:
            # Make the box bigger to avoid as much as possible 'outlier'. This
            # is important if the sites are defined at a high density: in this
            # case the bulk regions don't have to be close to 1 * n0 but can
            # be less. It's much more difficult to deal with outliers.  The
            # ideal solution would use images: implement 'looking across the
            # periodic boundaries' but that gets complicate when the box
            # rotates due to RMS fitting.
            smin = np.min(coord, axis=0) - padding
            smax = np.max(coord, axis=0) + padding
            BINS = fixedwidth_bins(delta, smin, smax)
            arange = np.transpose(np.vstack((BINS['min'], BINS['max'])))
            bins = BINS['Nbins']
            # create empty grid with the right dimensions (and get the edges)
            grid, edges = np.histogramdd(np.zeros((1, 3)), bins=bins, range=arange, normed=False)
            grid *= 0.0
        self._grid = grid
        self._edges = edges
        h = grid.copy()
        self._arange = arange
        self._bins = bins

    def _prepare(self):
        """
        Additional preparation to run
        """

        # pm = ProgressMeter(self._n_frames, interval=interval,
        #                    verbose=verbose,
        #                    format="Histogramming %(n_atoms)6d atoms in frame "
        #                    "%(step)5d/%(numsteps)d  [%(percentage)5.1f%%]\r")
        # start, stop, step = u.trajectory.check_slice_indices(start, stop, step)

    def _single_frame(self, ts, atomgroups):
        """
        Performs computation on single trajectory frame.

        Creates a histogram of positions of atoms in an atom group for a single frame.

        Parameters
        ----------
        ts : int
            current time step
        atomgroups : AtomGroup
            atom group for current block

        Returns
        -------

        """
        coord = self.current_coordinates(atomgroups[0], self._cutoff, self._atomselection, self._soluteselection, self._update_selection)
        h, edges = np.histogramdd(coord, bins=self._bins, range=self._arange, normed=False)
        h = np.array(h)
        return [h, edges]

    def _conclude(self):
        """
        """
        self._edges = self._results[0, 1]
        self._grid = self._results[:, 0].sum(axis=0)
        self._grid /= float(self._n_frames)
        metadata = self._metadata if self._metadata is not None else {}
        # metadata['psf'] = u.filename
        metadata['dcd'] = self._trajectory.filename
        metadata['atomselection'] = self._atomselection
        metadata['n_frames'] = self._n_frames
        metadata['totaltime'] = round(self._n_frames * self._trajectory.dt, 3)
        metadata['dt'] = self._trajectory.dt
        metadata['time_unit'] = mda.core.flags['time_unit']
        try:
            metadata['trajectory_skip'] = self._trajectory.skip_timestep  # frames
        except AttributeError:
            metadata['trajectory_skip'] = 1
        try:
            metadata['trajectory_delta'] = self._trajectory.delta  # in native units
        except AttributeError:
            metadata['trajectory_delta'] = 1
        if self._cutoff > 0 and self._soluteselection is not None:
            metadata['soluteselection'] = self._soluteselection
            metadata['cutoff'] = self._cutoff  # in Angstrom

        parameters = self._parameters if self._parameters is not None else {}
        parameters['isDensity'] = False  # must override
        g = Density(grid=self._grid, edges=self._edges, units={'length': mda.core.flags['length_unit']},
                    parameters=parameters, metadata=metadata)
        g.make_density()
        self.g = g
        # logger.info("Density completed (initial density in Angstrom**-3)")

    @staticmethod
    def _reduce(res, result_single_frame):
        """ 'append' action for a time series"""
        if isinstance(res, list) and len(res) == 0:
            res = result_single_frame
        else:
            res[0] += result_single_frame[0]
        return res
    @staticmethod
    def current_coordinates(atomgroup, cutoff, atomselection, soluteselection, update_selection):
        u = atomgroup.universe
        if cutoff > 0 and soluteselection is not None:
        # special fast selection for '<atomsel> not within <cutoff> of <solutesel>'
            notwithin_coordinates = notwithin_coordinates_factory(u, atomselection, soluteselection, cutoff, use_kdtree=use_kdtree, updating_selection=update_selection)
            return notwithin_coordinates()
        else:
            group = u.select_atoms(atomselection, updating=update_selection)
            return group.positions
