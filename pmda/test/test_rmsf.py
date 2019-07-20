import pytest
import MDAnalysis as mda
import numpy as np
import pmda.rmsf
from MDAnalysis.analysis.rms import RMSF
from numpy.testing import assert_almost_equal
from MDAnalysisTests.datafiles import GRO_MEMPROT, XTC_MEMPROT

@pytest.fixture(scope='module')
def u():
    return mda.Universe(GRO_MEMPROT, XTC_MEMPROT)

def test_rmsf_sum(u):
    PMDA = pmda.rmsf.RMSF(u.atoms)
    core_number = 1
    PMDA.run(n_blocks=core_number, n_jobs=core_number)
    MDA = mda.analysis.rmsf.RMSF(u.atoms).run()
    assert np.sum(MDA.rmsf) == np.sum(PMDA.rmsf)

def test_density_grid(u):
    PMDA = pmda.rmsf.RMSF(u.atoms)
    core_number = 1
    PMDA.run(n_blocks=core_number, n_jobs=core_number)
    MDA = mda.analysis.rmsf.RMSF(u.atoms).run()
    assert_almost_equal(MDA.rmsf, PMDA.rmsf)
