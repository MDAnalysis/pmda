==============================================
  PMDA - Parallel Molecular Dynamics Analysis
==============================================

|build| |cov| |PRwelcome| |zenodo|

Ready to use analysis and buildings blocks to write parallel analysis algorithms
using MDAnalysis_ with dask_.

.. warning::
   This project is **alpha software** and not API stable. It will and
   should rapidly evolve to test different approaches to implementing
   parallel analysis in a seamless and intuitive fashion.


For example, run a rmsd analysis on all available cores:

.. code:: python

   import MDAnalysis as mda
   from pmda import rms

   u = mda.Universe(top, traj)
   ref = mda.Universe(top, traj)

   rmsd_ana = rms.RMSD(u.atoms, ref.atoms).run(n_jobs=-1)

   print(rmsd_ana.rmsd)


By default PMDA use the multiprocessing scheduler of dask_. This is sufficient
if you want to run your simulation on a single machine. If your analysis takes
a very long time (>30 min) you can also spread it to several nodes using the
distributed_ scheduler. To do this you can pass a ``scheduler`` keyword
argument to the ``run`` method.

To write your own parallel algorithms you can subclass the
``pmda.parallel.ParallelAnalysisBase`` class.


License and source code
=======================

PMDA is released under the `GNU General Public License, version 2`_ (see the
files AUTHORS and LICENSE for details).

Source code is available in the public GitHub repository
https://github.com/MDAnalysis/pmda/.

       
Installation
============

Install a release with ``pip``
------------------------------

The latest release is available from https://pypi.org/project/pmda/
and can be installed with pip_

.. code-block:: sh

   pip install --upgrade pmda
		

   
Development version from source
-------------------------------

To install the latest development version from source, run

.. code-block:: sh

  git clone git@github.com:MDAnalysis/pmda.git
  cd pmda
  python setup.py install

 
Getting help
============

*Help* is also available through the *MDAnalysis mailing list*

     https://groups.google.com/group/mdnalysis-discussion

Please report *bugs and feature requests* for PMDA through the `Issue
Tracker`_.



Contributing
============

PMDA welcomes new contributions. Please drop by the `MDAnalysis developer
mailing list`_ to discuss and ask questions.

To contribute code, submit a *pull request* against the master branch in the
`PMDA repository`_.


Citation
========

If you use PMDA in published work please cite [Linke2018]_.

.. [Linke2018] Max Linke, & Oliver Beckstein. (2018, May 11). MDAnalysis/pmda:
               0.1.0 (Version 0.1.0). Zenodo. http://doi.org/10.5281/zenodo.1245759

	       
.. _MDAnalysis: https://www.mdanalysis.org
.. _dask: https://dask.pydata.org/en/latest/
.. _distributed: https://distributed.readthedocs.io/
.. _`Issue tracker`: https://github.com/MDAnalysis/pmda/issues
.. _`PMDA repository`: https://github.com/MDAnalysis/pmda/
.. _pip: https://pip.pypa.io/en/stable/
.. _`GNU General Public License, version 2`:
   https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
.. _`MDAnalysis developer mailing list`:
   https://groups.google.com/group/mdnalysis-devel

.. |build| image:: https://travis-ci.org/MDAnalysis/pmda.svg?branch=master
    :alt: Build Status
    :target: https://travis-ci.org/MDAnalysis/pmda

.. |cov| image:: https://codecov.io/gh/MDAnalysis/pmda/branch/master/graph/badge.svg
   :alt: Coverage
   :target: https://codecov.io/gh/MDAnalysis/pmda

.. |zenodo| image:: https://zenodo.org/badge/106346721.svg
   :alt: DOI
   :target: https://zenodo.org/badge/latestdoi/106346721

.. |PRwelcome| image:: https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square
   :alt: PRs welcome
   :target: http://makeapullrequest.com
