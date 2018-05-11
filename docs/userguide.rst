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

.. toctree::
   :maxdepth: 1

   userguide/pmda_classes
   userguide/parallelization
   userguide/new_classes
   
