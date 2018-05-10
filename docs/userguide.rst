.. -*- coding: utf-8 -*-

============
 User Guide
============

:mod:`pmda` implements a simple map-reduce scheme for parallel
trajectory analysis [Khoshlessan2017]_ for MDAnalysis [Gowers2016]_
[Michaud-Agrawal2011]_. The trajectory is partitioned into blocks and
analysis is performed separately and in parallel on each block
("map"). The results from each block are gathered and combined
("reduce"). :mod:`pmda` contains a number of pre-defined analysis
classes (see :ref:`pre-defined-analysis-tasks`) that are modelled
after functionality in :mod:`MDAnalysis.analysis` and that can be used
right away. However, often it can be almost as easy to :ref:`write
your own parallel class<example-new-parallel-analysis>`.


Using the :mod:`pmda` analysis classes
======================================

In order to use the parallel RMSD calculator :class:`pmda.rms.RMSD`,
import the module, set up two
:class:`~MDAnalysis.core.groups.AtomGroup` instances, instantiate the
class and directly use its :meth:`~pmda.rms.RMSD.run` method:

.. code:: python

   import MDAnalysis as mda
   from pmda import rms

   u = mda.Universe(top, traj)
   ref = mda.Universe(top, traj)

   rmsd_ana = rms.RMSD(u.atoms, ref.atoms).run(n_jobs=-1)

   print(rmsd_ana.rmsd)

The resulting ``rmsd_ana`` instance contains the array of the RMSD
time series in the attribute :attr:`~pmda.rms.RMSD.rmsd`.

Under the hood, Dask_ is used for the parallelization. By
default, all the available cores on the local machine (laptop or
workstation) are used with the ``n_jobs=-1`` keyword but any number
can be set, e.g., ``n_jobs=4`` to split the trajectory into 4 blocks.

One can also supply a `dask.distributed`_ scheduler in the ``get``
keyword argument. This makes it possible to run analysis in a
distributed fashion on HPC or ad-hoc clusters.

- TODO: example for launching distributed
- TODO: show ``run(..., get=scheduler)``  

.. _Dask: https://dask.pydata.org
.. _`dask.distributed`:  https://distributed.readthedocs.io/


.. _example-new-parallel-analysis:

Example: Writing new parallel analysis
======================================

With the help of :class:`pmda.parallel.ParallelAnalysisBase` one can
write new analysis functions that automatically parallelize.

1. Define the *single frame* analysis function, i.e., how to compute
   the observable for a single time step from a given
   :class:`~MDAnalysis.core.groups.AtomGroup`.
2. Derive a class from :class:`~pmda.parallel.ParallelAnalysisBase`
   that uses the single frame function.

As an example, we show how one can parallelize the RMSF function (from
:class:`MDAnalysis.analysis.rms.RMSF`):

- TODO       
- more TODO
- other example?  
