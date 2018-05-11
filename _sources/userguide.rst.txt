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

Pre-defined analysis classes follow a common pattern. They require one
or more MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup`
instances as input (which one obtains from a MDAnalysis
:class:`~MDAnalysis.core.universe.Universe`, typically by performing a
`selection`_). They also typically require a number of parameters that
are provided as keyword arguments:

.. class:: ParallelAnalysisClass(atomgroup[, atomgroup[, ...]], **kwargs)

	   set up the parallel analysis
	   
   .. method:: run(n_jobs=-1, scheduler=None)
	       
	       perform parallel analysis; see :ref:`parallelization`
	       for explanation of the arguments

   .. attribute:: results

	       stores the results (name and content depends on the
	       analysis that is being performed)

The instance of the parallel analysis class contains a
:meth:`~ParallelAnalysisClass.run` method. Calling
:meth:`~ParallelAnalysisClass.run` *performs* the parallel
analysis. Using keyword arguments, one can set the level of
:ref:`parallelization<parallelization>`.

After the run has been completed, results are stored in an attribute
of the class, such as :attr:`~ParallelAnalysisClass.results`. The
actual name of the attribute is defined by the specific analysis
class. The format and content is also dependent on the analysis; for
example, the RMSD analysis class :class:`pmda.rms.RMSD` produces a
numpy array of the RMSD time series.

.. _selection:
   https://www.mdanalysis.org/docs/documentation_pages/selections.html
   

.. _example-parallel-rmsd:

Example: Parallel RMSD
----------------------
	    
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


.. _parallelization:

Parallelization
===============

Under the hood, Dask_ is used for the parallelization. By
default, all the available cores on the local machine (laptop or
workstation) are used with the ``n_jobs=-1`` keyword but any number
can be set, e.g., ``n_jobs=4`` to split the trajectory into 4 blocks.

One can also supply a `dask.distributed`_ scheduler in the
``scheduler`` keyword argument of the
:meth:`~pmda.parallel.ParallelAnalysisBase.run` method. This makes it
possible to run analysis in a distributed fashion on HPC or ad-hoc
clusters (see `setting up a dask.distributed network`_).

.. _Dask: https://dask.pydata.org
.. _`dask.distributed`:  https://distributed.readthedocs.io/
.. _`setting up a dask.distributed network`:
   https://distributed.readthedocs.io/en/latest/setup.html


Example: Using ``dask.distributed``
-----------------------------------

You can try out `dask.distributed`_ with a `local cluster`_, which
sets up a scheduler and workers on the local machine. 

.. code:: python

   import distributed
   lc = distributed.LocalCluster(n_workers=8, processes=True)
   client = distributed.Client(lc)

The ``client`` can be passed to the ``scheduler`` argument of the
:meth:`~pmda.parallel.ParallelAnalysisBase.run` method; we continue to
use the :ref:`RMSD example<example-parallel-rmsd>`):
      
.. code:: python

   rmsd_ana = rms.RMSD(u.atoms, ref.atoms).run(scheduler=client)	  

Because the local cluster contains 8 workers, the RMSD trajectory
analysis will be parallelized over 8 trajectory segments.

In order to run on a larger cluster with multiple nodes (see `setting
up a dask.distributed network`_) one needs to know how to connect to
the running scheduler (e.g., address and port number or shared state
file). Assuming that the scheduler is running on 192.168.0.1:8786, one
would initialize the `distributed.Client`_ and pass it to the parallel
analysis :meth:`~pmda.parallel.ParallelAnalysisBase.run` method:

.. code:: python

   import distributed
   client = distributed.Client('192.168.0.1:8786')   
   rmsd_ana = rms.RMSD(u.atoms, ref.atoms).run(scheduler=client)	  

In this way one can spread an analysis task over many different nodes.
   
.. _`local cluster`:
   https://distributed.readthedocs.io/en/latest/local-cluster.html
.. _`distributed.Client`:
   https://distributed.readthedocs.io/en/latest/client.html

   
.. _example-new-parallel-analysis:

Writing new parallel analysis
=============================

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
