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


def test_rmsf_sum(u):
    PMDA = pmda.rmsf.RMSF(u.atoms)
    core_number = 1
    PMDA.run(n_blocks=core_number, n_jobs=core_number)
    MDA = mda.analysis.rms.RMSF(u.atoms).run()
    assert np.sum(MDA.rmsf) == np.sum(PMDA.rmsf)


def test_rmsf_values(u):
    PMDA = pmda.rmsf.RMSF(u.atoms)
    core_number = 1
    PMDA.run(n_blocks=core_number, n_jobs=core_number)
    MDA = mda.analysis.rms.RMSF(u.atoms).run()
    assert_almost_equal(MDA.rmsf, PMDA.rmsf)


@pytest.fixture
def setup(tmpdir):
    newdir = tmpdir.mkdir('resources')
    return newdir.dirname


def test_nottmp():
    filepath = os.path.join(os.path.realpath(pmda.__file__))
    assert os.path.exists(filepath)


def test_tmpdir(setup):
    os.chdir(setup)
    filepath = os.path.join(os.path.realpath(pmda.__file__))
    assert os.path.exists(filepath)
