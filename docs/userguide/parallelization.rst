.. -*- coding: utf-8 -*-

.. _parallelization:

=================
 Parallelization
=================

Dask_ is used to parallelize analysis in PMDA. It provides a flexible
approach to task-based parallelism and can scale from multi-core
laptops to large compute clusters.


Single machine
==============

By default, all the available cores on the local machine (laptop or
workstation) are used with the ``n_jobs=-1`` keyword but any number
can be set, e.g., ``n_jobs=4`` to split the trajectory into 4 blocks.

Internally, this uses the processes (multiprocessing) `scheduler`_
of dask. If you want to make use of more advanced scheduler features
or scale your analysis to multiple nodes, e.g., in an HPC (high
performance computing) environment, then use the :mod:`distributed`
scheduler, as described next. If ``n_jobs==1`` a synchronous
(single threaded) scheduler is used [#threads]_.

.. _`scheduler`:
   https://docs.dask.org/en/latest/scheduler-overview.html


``dask.distributed``
====================

With the `distributed`_ scheduler on can run analysis in a distributed
fashion on HPC or ad-hoc clusters (see `setting up a dask.distributed
network`_) or on a `single machine`_. (In addition, *distributed* also
provides `diagnostics`_ in form of a dashboard in the browser and a
progress bar.)

.. _Dask: https://dask.org
.. _`distributed`:  https://distributed.readthedocs.io/
.. _`setting up a dask.distributed network`:
   https://distributed.readthedocs.io/en/latest/setup.html
.. _`single machine`:
   http://docs.dask.org/en/latest/setup/single-distributed.html
.. _diagnostics:
   http://docs.dask.org/en/latest/diagnostics-distributed.html

Local cluster (single machine)
------------------------------

You can try out `dask.distributed`_ with a `local cluster`_, which
sets up a scheduler and workers on the local machine.

.. code:: python

   import distributed
   lc = distributed.LocalCluster(n_workers=8, processes=True)
   client = distributed.Client(lc)

Setting up the ``client`` is sufficient for Dask_ (and PMDA, namely the
:meth:`~pmda.parallel.ParallelAnalysisBase.run` method) to use it. We
continue to use the :ref:`RMSD example<example-parallel-rmsd>`:

.. code:: python

   rmsd_ana = rms.RMSD(u.atoms, ref.atoms).run()

Because the local cluster contains 8 workers, the RMSD trajectory
analysis will be parallelized over 8 trajectory segments.


Cluster
-------

In order to run on a larger cluster with multiple nodes (see `setting
up a dask.distributed network`_) one needs to know how to connect to
the running scheduler (e.g., address and port number or shared state
file). Assuming that the scheduler is running on 192.168.0.1:8786, one
would initialize the `distributed.Client`_ and this is enough to use
*distributed* for all analysis (it `configures the scheduler`_ to be
*distributed*):

.. code:: python

   import distributed
   client = distributed.Client('192.168.0.1:8786')
   rmsd_ana = rms.RMSD(u.atoms, ref.atoms).run()

In this way one can spread an analysis task over many different nodes.

.. _`local cluster`:
   https://distributed.readthedocs.io/en/latest/local-cluster.html
.. _`distributed.Client`:
   https://distributed.readthedocs.io/en/latest/client.html
.. _`configures the scheduler`:
   https://docs.dask.org/en/latest/scheduling.html#configuration

.. rubric:: Footnotes
.. [#threads] The *synchronous* scheduler is very useful for
	      debugging_. By setting ``n_jobs=1`` and not using a
	      *distributed* scheduler, the synchronous scheduler is
	      automatically used. Alternatively, set the synchronous
        scheduler with

	      .. code:: python

	         dask.config.set(scheduler='synchronous')

	      for any ``n_jobs``.

.. _debugging:
   https://docs.dask.org/en/latest/scheduling.html#single-thread
