# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version

import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysisTests.datafiles import DCD, PSF
import joblib

from dask import distributed, multiprocessing

from pmda import parallel


def test_timeing():
    io = np.arange(5)
    compute = np.arange(5) + 1
    total = 5
    universe = np.arange(2)
    timing = parallel.Timing(io, compute, total, universe)

    np.testing.assert_equal(timing.io, io)
    np.testing.assert_equal(timing.compute, compute)
    np.testing.assert_equal(timing.total, total)
    np.testing.assert_equal(timing.universe, universe)
    np.testing.assert_equal(timing.cumulate_time, np.sum(io) + np.sum(compute))


class NoneAnalysis(parallel.ParallelAnalysisBase):
    def __init__(self, atomgroup):
        universe = atomgroup.universe
        super(NoneAnalysis, self).__init__(universe, (atomgroup, ))

    def _prepare(self):
        pass

    def _conclude(self):
        self.res = np.hstack(self._results)

    def _single_frame(self, ts, atomgroups):
        return ts.frame


@pytest.fixture
def analysis():
    u = mda.Universe(PSF, DCD)
    ana = NoneAnalysis(u.atoms)
    return ana


def test_wrong_scheduler(analysis):
    with pytest.raises(ValueError):
        analysis.run(scheduler=2)


@pytest.mark.parametrize('n_jobs', (1, 2))
def test_all_frames(analysis, n_jobs):
    analysis.run(n_jobs=n_jobs)
    u = mda.Universe(analysis._top, analysis._traj)
    assert len(analysis.res) == u.trajectory.n_frames


@pytest.mark.parametrize('n_jobs', (1, 2))
def test_sub_frames(analysis, n_jobs):
    analysis.run(start=10, stop=50, step=10, n_jobs=n_jobs)
    np.testing.assert_almost_equal(analysis.res, [10, 20, 30, 40])


@pytest.fixture(scope="session")
def client(tmpdir_factory):
    with tmpdir_factory.mktemp("dask_cluster").as_cwd():
        lc = distributed.LocalCluster(n_workers=2, processes=True)
        client = distributed.Client(lc)

        yield client

        client.close()
        lc.close()


@pytest.fixture(scope='session', params=('distributed', 'multiprocessing'))
def scheduler(request, client):
    if request.param == 'distributed':
        return client
    else:
        return multiprocessing


def test_scheduler(analysis, scheduler):
    analysis.run(scheduler=scheduler)


@pytest.mark.parametrize('n_blocks', np.arange(1, 11))
def test_nblocks(analysis, n_blocks):
    analysis.run(n_blocks=n_blocks)
    assert len(analysis._results) == n_blocks


def test_guess_nblocks(analysis):
    analysis.run(n_jobs=-1)
    assert len(analysis._results) == joblib.cpu_count()
