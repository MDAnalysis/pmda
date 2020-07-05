.. pmda documentation master file, created by
   sphinx-quickstart on Tue May 23 16:00:46 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
	     
==============================================
  PMDA - Parallel Molecular Dynamics Analysis
==============================================

|build| |cov| |PRwelcome| |zenodo|

:Release: |release|
:Date: |today|

Ready to use analysis and buildings blocks to write parallel analysis algorithms
using MDAnalysis_ with Dask_.

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


By default PMDA uses the multiprocessing scheduler of Dask_. This is sufficient
if you want to run your simulation on a single machine although you may use any
of the `single-machine schedulers`_ that Dask supports. If your analysis takes
a long time you can also spread it to several nodes using the distributed_
scheduler (see :ref:`parallelization` for more details).

To write your own parallel algorithms you can subclass the
:class:`~pmda.parallel.ParallelAnalysisBase` class (see
:ref:`example-new-parallel-analysis` for more details).

     
License and source code
=======================

PMDA is released under the `GNU General Public License, version 2`_ (see the
files AUTHORS and LICENSE for details).

Source code is available in the public GitHub repository
https://github.com/MDAnalysis/pmda/.

       
Installation
============

The easiest way to install PMDA is as a :ref:`conda package <install-conda>`
and will result in a complete installation on Linux, macOS, and Windows
[#windows-support]_. If you have not installed Python packages before, we
recommend that you perform the conda installation.

PMDA runs under Python 3.4 or higher or Python 2.7 on Linux, macOS, and
Windows [#windows-support]_.


.. _install-conda:

Install a release with ``conda``
--------------------------------

First installation with conda_:	

.. code-block:: bash 

   conda config --add channels conda-forge
   conda install pmda

which will automatically install a *full set of dependencies* (including
MDAnalysis_, Dask_, and distributed_).

To upgrade later:

.. code-block:: bash 

   conda update pmda




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

*Help* is also available through the `MDAnalysis mailing list`_.

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

If you use PMDA in published work please cite [Fan2019]_.
	       
	       
.. _MDAnalysis: https://www.mdanalysis.org
.. _Dask: https://dask.org/
.. _`single-machine schedulers`:
   https://docs.dask.org/en/latest/setup/single-machine.html
.. _distributed: https://distributed.readthedocs.io/
.. _`Issue tracker`: https://github.com/MDAnalysis/pmda/issues
.. _`PMDA repository`: https://github.com/MDAnalysis/pmda/
.. _conda: https://conda.io/docs/
.. _pip: https://pip.pypa.io/en/stable/
.. _`GNU General Public License, version 2`:
   https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
.. _`MDAnalysis mailing list`:
   https://groups.google.com/group/mdnalysis-discussion   
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


.. Hide the contents from the front page because they are already in
.. the side bar in the Alabaster sphinx style; requires Alabaster
.. config sidebar_includehidden=True (default)

.. Contents
.. ========

.. toctree::
   :maxdepth: 4
   :hidden:      

   userguide
   api   
   references


.. rubric:: Footnotes
.. [#windows-support] PMDA itself is a pure Python package and runs without
   problems on all operating systems that support Python. However, it requires
   MDAnalysis for its operation and support for Windows is currently
   *experimental in MDAnalysis 0.19.1*.
