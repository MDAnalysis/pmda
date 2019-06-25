import numpy as np
import MDAnalysis as mda
import dask
from dask.delayed import delayed
import time
from .parallel import ParallelAnalysisBase
from .util import timeit, make_balanced_slices

class RMSF(ParallelAnalysisBase):
    """Parallel RMSF analysis.

    Attributes
    ----------
    Parameters
    ----------

    Note
    ----
    At the moment, this class doesn't do anything.
    """

    def __init__(self, atomgroup):
        """
        Parameters
        ----------
        atomgroup : AtomGroup
            atom groups that are iterated in parallel
        """
        u = atomgroup.universe
        super(RMSF, self).__init__(u, (atomgroup,))
        self._atomgroup = atomgroup
        self._top = u.filename
        self._traj = u.trajectory.filename

    def _prepare(self):
        """Additional preparation to run"""
        self.sumsquares = np.zeros((self._atomgroup.n_atoms, 3))
        self.mean = np.zeros((self._atomgroup.n_atoms, 3))

    def _single_frame(self, ts, agroups, frame_index, sumsquares, mean):
        """
        Performs computation on single trajectory frame.

        Parameters
        ----------
        ts : int
            current time step
        agroups : AtomGroup
            atom group for current block
        frame_index : int
            current frame index
        sumsquares : int
            current sum of squares of positions
        mean : array
            current running mean position

        Returns
        -------
        sumsquares : array
            updated sum of squares
        mean : array
            updated mean
        """
        k = frame_index
        sumsquares += (k / (k+1)) * (agroups.positions - mean) ** 2
        mean = (k * mean + agroups.positions) / (k + 1)
        return sumsquares, mean

    def run(self,
            start=None,
            stop=None,
            step=None,
            n_jobs=-1,
            n_blocks=1):

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
            number of tasks to start, if `-1` use number of logical cpu cores.
            This argument will be ignored when the distributed scheduler is
            used
        n_blocks : int, optional
            number of blocks to divide trajectory into. If ``None`` set equal
            to n_jobs or number of available workers in scheduler.
        """
        # are we using a distributed scheduler or should we use
        # multiprocessing?
        scheduler = dask.config.get('scheduler', None)
        if scheduler is None:
            # maybe we can grab a global worker
            try:
                scheduler = dask.distributed.worker.get_client()
            except ValueError:
                pass

        if n_jobs == -1:
            n_jobs = cpu_count()

        # we could not find a global scheduler to use and we ask for a single
        # job. Therefore we run this on the single threaded scheduler for
        # debugging.
        if scheduler is None and n_jobs == 1:
            scheduler = 'single-threaded'

        # fall back to multiprocessing, we tried everything
        if scheduler is None:
            scheduler = 'multiprocessing'

        if n_blocks is None:
            if scheduler == 'multiprocessing':
                n_blocks = n_jobs
            elif isinstance(scheduler, dask.distributed.Client):
                n_blocks = len(scheduler.ncores())
            else:
                n_blocks = 1
                warnings.warn(
                    "Couldn't guess ideal number of blocks from scheduler."
                    "Setting n_blocks=1. "
                    "Please provide `n_blocks` in call to method.")

        scheduler_kwargs = {'scheduler': scheduler}
        if scheduler == 'multiprocessing':
            scheduler_kwargs['num_workers'] = n_jobs

        self._indices = self._atomgroup.indices
        self._ind_slices = np.array_split(self._indices, n_blocks)

        with timeit() as total:
            with timeit() as prepare:
                self._prepare()
            start, stop, step = self._trajectory.check_slice_indices(
                start, stop, step)
            self._frames = range(start, stop, step)
            blocks = []
            for ind_slice in self._ind_slices:
                task = delayed(
                    self._dask_helper, pure=False)(
                             ind_slice,
                             self._top,
                             self._traj,
                             self._frames)
                blocks.append(task)
            blocks = delayed(blocks)
            res = blocks.compute(**scheduler_kwargs)
            self._results = np.hstack(res)
            with timeit() as conclude:
                self._conclude()
        return self

    def _conclude(self):
        """
        Takes the root-mean of the final sum of squares from each atom in the
        trajectory.
        """
        k = len(self._frames)
        self.sumsquares = self._results[0]
        self.mean = self._results[1]
        self.rmsf = np.sqrt(self._results[0].sum(axis=1) / k)

    def _dask_helper(self, ind_slice, top, traj, frames):
        """helper function to actually setup dask graph"""
        # wait_end needs to be first line for accurate timing
        wait_end = time.time()
        with timeit() as b_universe:
            u = mda.Universe(top, traj)
            agroups = u.atoms[ind_slice]

        sumsquares = np.zeros((agroups.n_atoms, 3))
        mean = np.zeros((agroups.n_atoms, 3))
        for i, frame in enumerate(frames):
            with timeit() as b_io:
                ts = u.trajectory[frame]
                frame_index = i
                sumsquares, mean = self._single_frame(ts, agroups, frame_index,
                sumsquares, mean)
        return sumsquares, mean

        def _reduce(res, result_single_frame):
            """ 'append' action for a time series"""
            res.append(result_single_frame)
            return res
