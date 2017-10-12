.. -*- coding: utf-8 -*-
   
===============================
 API for the Parallel Analysis
===============================

:mod:`pmda` implements a simple map-reduce scheme for parallel
trajectory analysis [Khoslessan2017]_ for MDAnalysis [Gowers2016]_
[Michaud-Agrawal2011]_. Building new parallel analysis classes is easy
with the :class:`pmda.parallel.ParallelAnalysisBase` base class if you
can formulate the problem as a calculation over one
:class:`~MDAnalysis.core.groups.AtomGroup` for a single frame.

.. autoclass:: pmda.parallel.ParallelAnalysisBase
  :members:

       
Example: Writing new parallel analysis
======================================

TODO (eg rmsd)


