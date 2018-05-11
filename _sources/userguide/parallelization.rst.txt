.. -*- coding: utf-8 -*-

.. _parallelization:

=================
 Parallelization
=================

Under the hood, Dask_ is used for the parallelization.

Single machine
==============

By default, all the available cores on the local machine (laptop or
workstation) are used with the ``n_jobs=-1`` keyword but any number
can be set, e.g., ``n_jobs=4`` to split the trajectory into 4 blocks.

Internally, this uses the multiprocessing `scheduler`_ of dask. If you
want to make use of more advanced scheduler features or scale your
analysis to multiple nodes, e.g., in an HPC (high performance
computing) environment, then use the :mod:`distributed` scheduler, as
described next.

.. _`scheduler`:
   https://dask.pydata.org/en/latest/scheduler-overview.html

     
``dask.distributed``
====================

One can supply a `dask.distributed`_ scheduler in the ``scheduler``
keyword argument of the
:meth:`~pmda.parallel.ParallelAnalysisBase.run` method. This makes it
possible to run analysis in a distributed fashion on HPC or ad-hoc
clusters (see `setting up a dask.distributed network`_).

.. _Dask: https://dask.pydata.org
.. _`dask.distributed`:  https://distributed.readthedocs.io/
.. _`setting up a dask.distributed network`:
   https://distributed.readthedocs.io/en/latest/setup.html


Local cluster (single machine)
------------------------------

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


Cluster
-------

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

   
