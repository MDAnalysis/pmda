import pytest

import MDAnalysis as mda
import numpy as np
import pmda.density
from MDAnalysis.analysis.density import density_from_Universe

from numpy.testing import assert_almost_equal

from MDAnalysisTests.datafiles import GRO_MEMPROT, XTC_MEMPROT


@pytest.fixture(scope='module')
def u():
    return mda.Universe(GRO_MEMPROT, XTC_MEMPROT)


def test_density(u):
    pmda = pmda.density.DensityAnalysis(u.atoms, atomselection='name OD1',
                                        updating=True)
    core_number = 2
    pmda.run(n_blocks=core_number, n_jobs=core_number)
    mda = mda.analysis.density.density_from_Universe(u,
                                                     atomselection='name OD1',
                                                     update_selection=True)
    assert np.sum(mda.grid) == np.sum(pmda.density.grid)
    assert_almost_equal(mda.grid, pmda.density.grid)
