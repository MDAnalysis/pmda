import numpy as np
import MDAnalysis as mda
# from mda.parallel import ParallelAnalysisBase

u = mda.Universe("trajectories/YiiP_system.pdb", "trajectories/YiiP_system_9ns_center.xtc")

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

    def __init__(self, atomgroup, parameter):
        self._ag = atomgroup
        super(NewAnalysis, self).__init__(atomgroup.universe,
                                          self._ag)

    def _single_frame(self, ts, agroups):
        """REQUIRED
        Called for every frame.

        Parameters
        ----------
        ``self``        -
        ``ts``          -contains the current time step
        ``agroups``     -is a tuple of atomgroups that are updated to the current frame.

        Return result of `some_function` for a single frame.
        """
        def RMSF_calc():
            raise NotImplementedError
        return RMSF_calc(agroups[0], self._parameter)

    def _conclude(self):
        # REQUIRED
        # Called once iteration on the trajectory is finished. Results
        # for each frame are stored in ``self._results`` in a per block
        # basis. Here those results should be moved and reshaped into a
        # sensible new variable.
        self.results = np.hstack(self._results)
        # Apply normalisation and averaging to results here if wanted.
        self.results /= np.sum(self.results

    @staticmethod
    def _reduce(res, result_single_frame):
        # NOT REQUIRED
        # Called for every frame. ``res`` contains all the results
        # before current time step, and ``result_single_frame`` is the
        # result of self._single_frame for the current time step. The
        # return value is the updated ``res``. The default is to append
        # results to a python list. This approach is sufficient for
        # time-series data.
        res.append(results_single_frame)
        # This is not suitable for every analysis. To add results over
        # multiple frames this function can be overwritten. The default
        # value for ``res`` is an empty list. Here we change the type to
        # the return type of `self._single_frame`. Afterwards we can
        # safely use addition to accumulate the results.
        if res == []:
            res = result_single_frame
        else:
            res += result_single_frame
        # If you overwrite this function *always* return the updated
        # ``res`` at the end.
        return res


run(start=None, stop=None, step=None, n_jobs=1, n_blocks=None)
na = NewAnalysis(u.select_atoms('name CA'), 35).run()
print(na.result)
