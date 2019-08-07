import pytest
import MDAnalysis as mda
import numpy as np
import os
import pytest
import pmda.rmsf
import MDAnalysis.analysis.rms
from numpy.testing import assert_almost_equal
from MDAnalysisTests.datafiles import PSF, DCD


@pytest.fixture(scope='module')
def u():
    return mda.Universe(PSF, DCD)


@pytest.mark.parametrize('n_cores', (1, 2, 3, 4, 5, 10))
@pytest.mark.parametrize('n_frames', (10, 50, 100))
def test_RMSF_values(u, n_cores, n_frames):
    PMDA_vals = pmda.rmsf.RMSF(u.atoms).run(stop=n_frames,
                                            n_blocks=n_cores,
                                            n_jobs=n_cores)
    MDA_vals = mda.analysis.rms.RMSF(u.atoms).run(stop=n_frames)
    assert_almost_equal(MDA_vals.mean, PMDA_vals.mean)
    assert_almost_equal(MDA_vals.sumsquares, PMDA_vals.sumsquares)
    assert_almost_equal(MDA_vals.rmsf, PMDA_vals.rmsf)


@pytest.mark.parametrize('n_cores', (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
def test_RMSF_n_jobs(u, n_cores):
    PMDA_vals = pmda.rmsf.RMSF(u.atoms).run(stop=10,
                                            n_blocks=n_cores,
                                            n_jobs=1)
    MDA_vals = mda.analysis.rms.RMSF(u.atoms).run(stop=10)
    assert_almost_equal(MDA_vals.mean, PMDA_vals.mean)
    assert_almost_equal(MDA_vals.sumsquares, PMDA_vals.sumsquares)
    assert_almost_equal(MDA_vals.rmsf, PMDA_vals.rmsf)


@pytest.mark.parametrize('rmsf_array', [pytest.param(np.array([-1, -1, -1]),
                                        marks=pytest.mark.xfail)])
def test_negative_rmsf(u, rmsf_array):
    PMDA_res = pmda.rmsf.RMSF(u.atoms).run(n_blocks=2, n_jobs=2)
    PMDA_res._negative_rmsf(np.array([0, -1, -2]))
