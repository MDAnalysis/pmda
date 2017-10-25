.. -*- coding: utf-8 -*-
   
=====================
 API for :mod:`pmda`
=====================

The :mod:`pmda` package implements a simple map-reduce scheme for
parallel trajectory analysis [Khoslessan2017]_ and extends MDAnalysis
[Gowers2016]_ [Michaud-Agrawal2011]_. In order to make use of it, a
user should be familiar with MDAnalysis_.

.. _MDAnalysis: https://www.mdanalysis.org


Building blocks
===============

Building new parallel analysis classes is easy with the
:class:`pmda.parallel.ParallelAnalysisBase` base class if you can
formulate the problem as a calculation over one
:class:`~MDAnalysis.core.groups.AtomGroup` for a single frame.

.. autoclass:: pmda.parallel.ParallelAnalysisBase
  :members:
  :private-members:
       

.. _pre-defined-analysis-tasks:

Pre-defined parallel analysis tasks
===================================


.. toctree::
   :maxdepth: 1

   api/rms
   api/contacts
