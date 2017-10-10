==============================================
  PMDA - Parallel Molecular Dynamics Analysis
==============================================

Ready to use analysis and buildings blocks to write parallel analysis algorithms
using MDAnalysis_.

.. code:: python

   import MDAnalysis as mda
   from pmda import rms

   u = mda.Universe(top, traj)
   ref = mda.Universe(top, traj)

   rmsd_ana = rms.RMSD(u.atoms, ref.atoms).run(n_jobs=-1)

   print(rmsd_ana.rmsd)


.. _MDAnalysis: http://www.mdanalysis.org
