# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
from __future__ import absolute_import

import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysisTests.datafiles import DCD, PSF
import joblib

import dask

from pmda import parallel
from pmda.util import make_balanced_slices


def test_timeing():
    io = np.arange(5)
    compute = np.arange(5) + 1
    total = 5
    prepare = 3
    prepare_dask = 4
    conclude = 6
    wait = 12
    io_block = np.sum(io)
    compute_block = np.sum(compute)

    timing = parallel.Timing(io, compute, total,
                             prepare, prepare_dask, conclude, wait,
                             io_block, compute_block,)

    np.testing.assert_equal(timing.io, io)
    np.testing.assert_equal(timing.compute, compute)
    np.testing.assert_equal(timing.total, total)
    np.testing.assert_equal(timing.cumulate_time, np.sum(io) + np.sum(compute))
    np.testing.assert_equal(timing.prepare, prepare)
    np.testing.assert_equal(timing.prepare_dask, prepare_dask)
    np.testing.assert_equal(timing.conclude, conclude)
    np.testing.assert_equal(timing.wait, wait)
    np.testing.assert_equal(timing.io_block, io_block)
    np.testing.assert_equal(timing.compute_block, compute_block)


class NoneAnalysis(parallel.ParallelAnalysisBase):
    def __init__(self, atomgroup):
        universe = atomgroup.universe
        super().__init__(universe)
        self._atomgroup = atomgroup

    def _prepare(self):
        pass

    def _conclude(self):
        self.res = np.hstack(self._results)

    def _single_frame(self):
        return self._ts.frame


@pytest.fixture
def analysis():
    u = mda.Universe(PSF, DCD)
    ana = NoneAnalysis(u.atoms)
    return ana


@pytest.mark.parametrize('n_jobs', (1, 2))
def test_all_frames(analysis, n_jobs):
    analysis.run(n_jobs=n_jobs)
    u = analysis._universe
    assert len(analysis.res) == u.trajectory.n_frames


@pytest.mark.parametrize('n_jobs', (1, 2))
def test_sub_frames(analysis, n_jobs):
    analysis.run(start=10, stop=50, step=10, n_jobs=n_jobs)
    np.testing.assert_almost_equal(analysis.res, [10, 20, 30, 40])


@pytest.mark.parametrize('n_jobs', (1, 2, 3))
def test_no_frames(analysis, n_jobs):
    u = analysis._universe
    n_frames = u.trajectory.n_frames
    with pytest.warns(UserWarning):
        analysis.run(start=n_frames, stop=n_frames+1, n_jobs=n_jobs)
    assert len(analysis.res) == 0
    np.testing.assert_equal(analysis.res, [])
    np.testing.assert_equal(analysis.timing.compute, [])
    np.testing.assert_equal(analysis.timing.io, [])
    np.testing.assert_equal(analysis.timing.io_block, [0])
    np.testing.assert_equal(analysis.timing.compute_block, [0])
    np.testing.assert_equal(analysis.timing.wait, [0])


def test_scheduler(analysis, scheduler):
    analysis.run()


def test_nframes_less_nblocks_warning(analysis):
    u = analysis._universe
    n_frames = u.trajectory.n_frames
    with pytest.warns(UserWarning):
        analysis.run(stop=2, n_blocks=4, n_jobs=2)
    assert len(analysis.res) == 2


@pytest.mark.parametrize('n_blocks', np.arange(1, 11))
def test_nblocks(analysis, n_blocks):
    analysis.run(n_blocks=n_blocks)
    assert len(analysis._results) == n_blocks


def test_guess_nblocks(analysis):
    with dask.config.set(scheduler='processes'):
        analysis.run(n_jobs=-1)
    assert len(analysis._results) == joblib.cpu_count()


@pytest.mark.parametrize('n_blocks', np.arange(1, 11))
def test_blocks(analysis, n_blocks):
    analysis.run(n_blocks=n_blocks)
    u = analysis._universe
    n_frames = u.trajectory.n_frames
    start, stop, step = u.trajectory.check_slice_indices(
                            None, None, None)
    slices = make_balanced_slices(n_frames, n_blocks, start, stop, step)
    blocks = [
        range(bslice.start, bslice.stop, bslice.step) for bslice in slices
        ]
    assert analysis._blocks == blocks


def test_attrlock():
    u = mda.Universe(PSF, DCD)
    pab = parallel.ParallelAnalysisBase(u)

    # Should initially be allowed to set attributes
    pab.thing1 = 24
    assert pab.thing1 == 24
    # Apply lock
    with pab.readonly_attributes():
        # Reading should still work
        assert pab.thing1 == 24
        # Setting should fail
        with pytest.raises(AttributeError):
            pab.thing2 = 100
    # Outside of lock context setting should again work
    pab.thing2 = 100
    assert pab.thing2 == 100


def test_reduce():
    res = []
    u = mda.Universe(PSF, DCD)
    ana = NoneAnalysis(u.atoms)
    res = ana._reduce(res, [1])
    res = ana._reduce(res, [1])
    # Should see res become a list with 2 elements.
    assert res == [[1], [1]]
