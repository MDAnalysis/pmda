# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# PMDA
# Copyright (c) 2019 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version

import pytest
import MDAnalysis as mda
import numpy as np
import os
import pytest
from pmda.rms import RMSF
import MDAnalysis.analysis.rms
from numpy.testing import assert_almost_equal
from MDAnalysisTests.datafiles import PSF, DCD


@pytest.fixture(scope='module')
def u():
    return mda.Universe(PSF, DCD)


@pytest.mark.parametrize('n_blocks', (1, 2, 3, 4, 5, 10))
@pytest.mark.parametrize('n_frames', (10, 50, 100))
def test_RMSF_values(u, n_blocks, n_frames):
    PMDA_vals = RMSF(u.atoms).run(stop=n_frames,
                                  n_blocks=n_blocks,
                                  n_jobs=n_blocks)
    MDA_vals = mda.analysis.rms.RMSF(u.atoms).run(stop=n_frames)
    assert_almost_equal(MDA_vals.mean, PMDA_vals.mean)
    assert_almost_equal(MDA_vals.sumsquares, PMDA_vals.sumsquares)
    assert_almost_equal(MDA_vals.rmsf, PMDA_vals.rmsf)


@pytest.mark.parametrize('n_blocks', (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
def test_RMSF_n_jobs(u, n_blocks):
    PMDA_vals = RMSF(u.atoms).run(stop=10,
                                  n_blocks=n_blocks,
                                  n_jobs=1)
    MDA_vals = mda.analysis.rms.RMSF(u.atoms).run(stop=10)
    assert_almost_equal(MDA_vals.mean, PMDA_vals.mean)
    assert_almost_equal(MDA_vals.sumsquares, PMDA_vals.sumsquares)
    assert_almost_equal(MDA_vals.rmsf, PMDA_vals.rmsf)


def test_negative_RMSF_raises_ValueError():
    with pytest.raises(ValueError):
        RMSF._negative_rmsf(np.array([-1, -1, -1]))
