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


def test_density_sum(u):
    pdensity = pmda.density.DensityAnalysis(u.atoms, atomselection='name OD1',
                                            updating=True)
    core_number = 2
    pdensity.run(n_blocks=core_number, n_jobs=core_number)
    dens = mda.analysis.density.density_from_Universe(u,
                                                      atomselection='name OD1',
                                                      update_selection=True)
    assert np.sum(dens.grid) == np.sum(pdensity.density.grid)


@pytest.mark.parametrize("n_blocks", [1, 2, 3, 4])
def test_density_grid(u, n_blocks):
    pdensity = pmda.density.DensityAnalysis(u.atoms, atomselection='name OD1',
                                            updating=True)
    core_number = n_blocks
    pdensity.run(n_blocks=core_number, n_jobs=core_number)
    dens = mda.analysis.density.density_from_Universe(u,
                                                      atomselection='name OD1',
                                                      update_selection=True)
    assert_almost_equal(dens.grid, pdensity.density.grid)


def test_updating(u):
    with pytest.raises(ValueError):
        pdensity = pmda.density.DensityAnalysis(u.atoms,
                                                updating=True)


def test_atomselection(u):
    with pytest.raises(ValueError):
        pdensity = pmda.density.DensityAnalysis(u.atoms,
                                                atomselection='name OD1')


def test_gridcenter(u):
    aselect = 'name OD1'
    gridcenter = np.array([10, 10, 10])
    xdim = 190
    ydim = 200
    zdim = 210
    dens = mda.analysis.density.density_from_Universe(u,
                                                      atomselection=aselect,
                                                      update_selection=True,
                                                      gridcenter=gridcenter,
                                                      xdim=xdim,
                                                      ydim=ydim,
                                                      zdim=zdim)
    pdens = pmda.density.DensityAnalysis(u.atoms,
                                         atomselection=aselect,
                                         updating=True,
                                         gridcenter=gridcenter,
                                         xdim=xdim,
                                         ydim=ydim,
                                         zdim=zdim)
    core_number = 4
    pdens.run(n_blocks=core_number, n_jobs=core_number)
    assert_almost_equal(dens.grid, pdens.density.grid)
    assert_almost_equal(pdens._gridcenter, gridcenter)
    assert len(pdens.density.edges[0]) == xdim + 1
    assert len(pdens.density.edges[1]) == ydim + 1
    assert len(pdens.density.edges[2]) == zdim + 1


@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("stop", [5, 4])
@pytest.mark.parametrize("start", [0, 1])
def test_n_frames(u, start, stop, step):
    D = pmda.density.DensityAnalysis(u.atoms)
    D.run(start, stop, step)
    assert D._n_frames == len(range(start, stop, step))
