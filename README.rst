==============================================
  PMDA - Parallel Molecular Dynamics Analysis
==============================================

|build| |cov|

:Release: |release|
:Date: |today|

Ready to use analysis and buildings blocks to write parallel analysis algorithms
using MDAnalysis_ with dask_.

.. warning::
   This project is **alpha software** and not API stable. It will and
   should rapidly evolve to test different approaches to implementing
   parallel analysis in a seamless and intuitive fashion.


For example to running a rmsd analysis on all available cores:

.. code:: python

   import MDAnalysis as mda
   from pmda import rms

   u = mda.Universe(top, traj)
   ref = mda.Universe(top, traj)

   rmsd_ana = rms.RMSD(u.atoms, ref.atoms).run(n_jobs=-1)

   print(rmsd_ana.rmsd)


By default pmda is using the multiprocessing scheduler of dask_. This is
sufficient if you want to run your simulation on a single machine. If your
analysis takes a very long time (>30 min) you can also spread it to several
nodes using the distributed_ scheduler. To do this can pass a `scheduler` keyword
argument to the `run` method.

To write your own parallel algorithms you can subclass the
`ParallelAnalysisBase` class.


Installation
============

To install the latest development version from source, run

.. code-block:: sh

  git clone git@github.com:MDAnalysis/pmda.git
  cd pmda
  python setup.py install

Getting help
============

For help using this library, please drop by the `Github Issue tracker`__

.. _issuetracker: https://github.com/MDAnalysis/pmda/issues

__ issuetracker_

.. _MDAnalysis: https://www.mdanalysis.org
.. _dask: https://dask.pydata.org/en/latest/
.. _distributed: https://distributed.readthedocs.io/

.. |build| image:: https://travis-ci.org/MDAnalysis/pmda.svg?branch=master
    :alt: Build Status
    :target: https://travis-ci.org/MDAnalysis/pmda

.. |cov| image:: https://codecov.io/gh/MDAnalysis/pmda/branch/master/graph/badge.svg
   :alt: Coverage
   :target: https://codecov.io/gh/MDAnalysis/pmda
