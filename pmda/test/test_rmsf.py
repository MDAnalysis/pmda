import pytest
import MDAnalysis as mda
import numpy as np
import os
import pytest
import pmda
import pmda.rmsf
from MDAnalysis.analysis.rms import RMSF
from numpy.testing import assert_almost_equal
from MDAnalysisTests.datafiles import GRO_MEMPROT, XTC_MEMPROT


@pytest.fixture(scope='module')
def u():
    return mda.Universe(GRO_MEMPROT, XTC_MEMPROT)


@pytest.mark.parametrize('n_cores', (1, 2, 3, 4, 5))
@pytest.mark.parametrize('n_frames', (10, 100))
def test_rmsf_sum(u, n_cores, n_frames):
    PMDA = pmda.rmsf.RMSF(u.atoms)
    PMDA.run(stop=n_frames, n_blocks=n_cores, n_jobs=n_cores)
    MDA = mda.analysis.rms.RMSF(u.atoms).run(stop=n_frames)
    assert np.sum(MDA.rmsf) == np.sum(PMDA.rmsf)

@pytest.mark.parametrize('n_cores', (1, 2, 3, 4, 5))
@pytest.mark.parametrize('n_frames', (10, 100))
def test_rmsf_values(u, n_cores, n_frames):
    PMDA = pmda.rmsf.RMSF(u.atoms)
    PMDA.run(stop=n_frames, n_blocks=n_cores, n_jobs=n_cores)
    MDA = mda.analysis.rms.RMSF(u.atoms).run(stop=n_frames)
    assert_almost_equal(MDA.rmsf, PMDA.rmsf)
