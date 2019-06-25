import numpy as np
import MDAnalysis as mda
import pmda
import dask
from dask.delayed import delayed
import time
from joblib import cpu_count
from pmda.parallel import ParallelAnalysisBase
from pmda.util import timeit

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
        u = atomgroup.universe
        super(RMSF, self).__init__(u, (atomgroup,))
        self._atomgroup = atomgroup
        self._top = u.filename
        self._traj = u.trajectory.filename

    def _prepare(self):
        self.sumsquares = np.zeros((self._atomgroup.n_atoms, 3))
        self.mean = np.zeros((self._atomgroup.n_atoms, 3))

    def _single_frame(self, ts, agroups, frame_index, sumsquares, mean):
        """
        Called for every frame.

        Parameters
        ----------
        ts : int
            current time step
        agroups :
            tuple of atomgroups that are updated to the current frame

        Returns result of RMSF function for a single frame.
        """
        k = frame_index
        # self.sumsquares += (k / (k+1)) * (self.atomgroup.positions - self.mean) ** 2
        sumsquares += (k / (k+1)) * (agroups.positions - mean) ** 2
        mean = (k * mean + agroups.positions) / (k + 1)
        return sumsquares, mean

    def run(self,
                start=None,
                stop=None,
                step=None,
                n_jobs=-1,
                n_blocks=1,
                cutoff=15.0):
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
            self._results = res
            self.sumsquares = res[0][0]
            self.mean = res[0][1]
            with timeit() as conclude:
                self._conclude()
            # self.timing = Timing(times_io,
            #                      np.hstack(timings), total.elapsed,
            #                      b_universe.elapsed, prepare.elapsed,
            #                      conclude.elapsed)
            return self

    def _conclude(self):
        # Called once iteration on the trajectory is finished. Results
        # for each frame are stored in ``self._results`` in a per block
        # basis. Here those results should be moved and reshaped into a
        # sensible new variable.
        # Apply normalization and averaging to results here if wanted.
        k = len(self._frames)
        self.rmsf = np.sqrt(self.sumsquares.sum(axis=1) / k)


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
                sumsquares, mean = self._single_frame(ts, agroups, frame_index, sumsquares, mean)
        return sumsquares, mean
