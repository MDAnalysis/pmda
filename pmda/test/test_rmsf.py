import pytest
import MDAnalysis as mda
import numpy as np
import os
import pytest
import pmda.rmsf
import MDAnalysis.analysis.rms
from numpy.testing import assert_almost_equal
from MDAnalysisTests.datafiles import GRO_MEMPROT, XTC_MEMPROT


@pytest.fixture(scope='module')
def u():
    return mda.Universe(GRO_MEMPROT, XTC_MEMPROT)


@pytest.mark.parametrize('n_cores', (1, 2, 3, 4, 5))
# @pytest.mark.parametrize('n_frames', (5))
def test_RMSF_values(u, n_cores, n_frames=5):
    atoms = u.select_atoms("all")
    PMDA_vals = pmda.rmsf.RMSF(atoms).run(stop=n_frames,
                                          n_blocks=n_cores,
                                          n_jobs=n_cores)
    MDA_vals = mda.analysis.rms.RMSF(atoms).run(stop=n_frames)
    assert_almost_equal(MDA_vals.mean, PMDA_vals.mean)
    assert_almost_equal(MDA_vals.sumsquares, PMDA_vals.sumsquares)
    assert_almost_equal(MDA_vals.rmsf, PMDA_vals.rmsf)
