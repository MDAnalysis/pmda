#%%

# from __future__ import absolute_import
#
# import pytest

import MDAnalysis as mda
import numpy as np
import pmda.density
from MDAnalysis.analysis.density import density_from_Universe
#
# from numpy.testing import assert_almost_equal
#
# from MDAnalysisTests.datafiles import GRO_MEMPROT, XTC_MEMPROT

#%%

u9 = mda.Universe("/nfs/homes/nawtrey/Documents/PMDA/trajectories/YiiP_system.pdb", "/nfs/homes/nawtrey/Documents/PMDA/trajectories/YiiP_system_9ns_center.xtc")
OH2 = u9.select_atoms('name OH2')
pmda_density = pmda.density.DensityAnalysis(OH2)
core_number = 2
pmda_density.run(n_blocks=core_number, n_jobs=core_number)

#%%


# mda_density = mda.analysis.density.density_from_Universe(u9, 1, atomselection='name OH2')
# np.sum(mda_density.grid)
# print(np.sum(pmda_density.g.grid))
# from numpy.testing import assert_almost_equal
# assert_almost_equal(mda_density.grid, pmda_density.g.grid)

#=======================================================================================================
#%%== Tests ============================================================================================
#=======================================================================================================

# mda = {
#         100 : mda100,
#         1000 : mda1000,
#         10000 : mda10000,
#         20000 : mda20000,
#         30000 : mda30000,
#         40000 : mda40000,
#         50000 : mda50000,
#         65000 : mda65000
#         }
#
# pmda = {
#         100 : pmda100,
#         1000 : pmda1000,
#         10000 : pmda10000,
#         20000 : pmda20000,
#         30000 : pmda30000,
#         40000 : pmda40000,
#         50000 : pmda50000,
#         65000 : pmda65000
#         }
#
# dataset = 65000
#
# np.sum(mda[dataset].grid) == np.sum(pmda[dataset].g.grid)
# from numpy.testing import assert_almost_equal
# assert_almost_equal(mda[dataset].grid, pmda[dataset].g.grid)
