.. -*- coding: utf-8 -*-

.. _example-new-parallel-analysis:

===============================
 Writing new parallel analysis
===============================

At the core of PMDA is the idea that a common interface makes it easy
to create code that can be easily parallelized, especially if the
analysis can be split into independent work over multiple trajectory
segments (or "slices")  and a final step in which all data from the
trajectory slices is combined.

.. _wrapping-functions:

Wrapping existing analysis functions
====================================

If you already have a function that

1. takes one or more :class:`~MDAnalysis.core.groups.AtomGroup`
   instances as input
2. analyzes one frame in a trajectory and returns the result for this
   frame

then you can use the helper functions in :mod:`pmda.custom` to rapidly
create a parallel analysis class that follows the :ref:`common PMDA
API<pmda-basics>`.


Example: Parallelizing radius of gyration
-----------------------------------------

For example, we want to calculate the radius of gyration of a
protein. We first create the protein ``AtomGroup``:

.. code:: python

   import MDAnalysis as mda
   u = mda.Universe(topology, trajectory)
   protein = u.select_atoms("protein")

The the following function calculates the radius of gyration of a
protein given in ``AtomGroup`` ``ag`` :

.. code:: python

   def rgyr(ag):
       return ag.radius_of_gyration()

We can wrap :func:`rgyr` in :class:`pmda.custom.AnalysisFromFunction`

.. code:: python

   import pmda.custom
   parallel_rgyr = pmda.custom.AnalysisFromFunction(rgyr, u, protein)

Run the analysis on 8 cores and show the timeseries that was collected
in the :attr:`results` attribute:

.. code:: python

   parallel_rgyr.run(n_jobs=8)
   print(parallel_rgyr.results)

   

Building PMDA analysis classes
==============================

With the help of :class:`pmda.parallel.ParallelAnalysisBase` one can
write new analysis functions that automatically parallelize. This
approach provides more freedom than :ref:`wrapping-functions`
described above.

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
