.. -*- coding: utf-8 -*-
   
=====================
 API for :mod:`pmda`
=====================

The :mod:`pmda` package implements a simple map-reduce scheme for
parallel trajectory analysis [Khoshlessan2017]_ and extends MDAnalysis
[Gowers2016]_ [Michaud-Agrawal2011]_. In order to make use of it, a
user should be familiar with MDAnalysis_.

.. _MDAnalysis: https://www.mdanalysis.org


Building blocks
===============

Building new parallel analysis classes is easy with the
:class:`pmda.custom.AnalysisFromFunction` base class if you can formulate the
problem as a calculation over one :class:`~MDAnalysis.core.groups.AtomGroup` for
a single frame. If your need more flexibility you can use the
:class:`pmda.parallel.ParallelAnalysisBase`.

.. toctree::
   :maxdepth: 1

   api/parallel
   api/custom

.. _pre-defined-analysis-tasks:

Pre-defined parallel analysis tasks
===================================

:mod:`pmda` also has various predefined analysis tasks to use. They
also function as examples for how to implement your own functions with
:class:`pmda.parallel.ParallelAnalysisBase`.

.. toctree::
   :maxdepth: 1

   api/rms
   api/contacts
