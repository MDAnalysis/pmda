.. -*- coding: utf-8 -*-

.. _example-new-parallel-analysis:

===============================
 Writing new parallel analysis
===============================

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
