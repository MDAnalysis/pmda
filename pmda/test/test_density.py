import pytest
import MDAnalysis as mda
import numpy as np
from pmda.density import DensityAnalysis
from MDAnalysis.analysis import density as serial_density
from numpy.testing import assert_almost_equal
from MDAnalysisTests.datafiles import GRO_MEMPROT, XTC_MEMPROT
from MDAnalysisTests.datafiles import PSF_TRICLINIC, DCD_TRICLINIC


@pytest.fixture(scope='module')
def u1():
    return mda.Universe(GRO_MEMPROT, XTC_MEMPROT)


@pytest.fixture(scope='module')
def u2():
    return mda.Universe(PSF_TRICLINIC, DCD_TRICLINIC)


@pytest.mark.parametrize("n_blocks", [1, 2])
@pytest.mark.parametrize("stop", [4, 5])
@pytest.mark.parametrize("step", [1, 2])
def test_density_values(u1, n_blocks, stop, step):
    parallel = DensityAnalysis(u1.atoms, atomselection='name OD1',
                               updating=True)
    parallel.run(n_blocks=n_blocks, n_jobs=n_blocks, start=0, stop=stop,
                 step=step)
    ag = u1.select_atoms('name OD1', updating=True)
    serial = serial_density.DensityAnalysis(ag).run(
        start=0, stop=stop, step=step)
    assert np.sum(serial.density.grid) == np.sum(parallel.density.grid)
    assert_almost_equal(serial.density.grid, parallel.density.grid, decimal=16)


def test_updating(u1):
    with pytest.raises(ValueError):
        pdensity = DensityAnalysis(u1.atoms, updating=True)


def test_atomselection(u1):
    with pytest.raises(ValueError):
        pdensity = DensityAnalysis(u1.atoms, atomselection='name OD1')


def test_gridcenter(u1):
    gridcenter = np.array([10, 10, 10])
    xdim = 190
    ydim = 200
    zdim = 210
    ag = u1.select_atoms('name OD1', updating=True)
    serial = serial_density.DensityAnalysis(
        ag,
        gridcenter=gridcenter,
        xdim=xdim,
        ydim=ydim,
        zdim=zdim).run()
    parallel = DensityAnalysis(u1.atoms, atomselection='name OD1',
                               updating=True,
                               gridcenter=gridcenter,
                               xdim=xdim,
                               ydim=ydim,
                               zdim=zdim)
    core_number = 4
    parallel.run(n_blocks=core_number, n_jobs=core_number)
    assert_almost_equal(serial.density.grid, parallel.density.grid)
    assert_almost_equal(parallel._gridcenter, gridcenter)
    assert len(parallel.density.edges[0]) == xdim + 1
    assert len(parallel.density.edges[1]) == ydim + 1
    assert len(parallel.density.edges[2]) == zdim + 1


@pytest.mark.parametrize("start", [0, 1])
@pytest.mark.parametrize("stop", [5, 7, 10])
@pytest.mark.parametrize("step", [1, 2, 3])
def test_n_frames(u2, start, stop, step):
    pdensity = DensityAnalysis(u2.atoms).run(start, stop, step)
    assert pdensity.n_frames == len(range(start, stop, step))
