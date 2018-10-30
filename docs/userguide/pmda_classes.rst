.. -*- coding: utf-8 -*-

.. _pmda-basics:
   
========================================
 Using the :mod:`pmda` analysis classes
========================================

Pre-defined analysis classes follow a common pattern. They require one
or more MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup`
instances as input (which one obtains from a MDAnalysis
:class:`~MDAnalysis.core.universe.Universe`, typically by performing a
`selection`_). They also typically require a number of parameters that
are provided as keyword arguments:

.. class:: ParallelAnalysisClass(atomgroup[, atomgroup[, ...]], **kwargs)

	   set up the parallel analysis
	   
   .. method:: run(n_jobs=-1)
	       
	       perform parallel analysis; see :ref:`parallelization`
	       for explanation of the arguments

   .. attribute:: results

	       stores the results (name and content depends on the
	       analysis that is being performed)

The instance of the parallel analysis class contains a
:meth:`~ParallelAnalysisClass.run` method. Calling
:meth:`~ParallelAnalysisClass.run` *performs* the parallel
analysis. Using keyword arguments, one can set the level of
:ref:`parallelization<parallelization>`.

After the run has been completed, results are stored in an attribute
of the class, such as :attr:`~ParallelAnalysisClass.results`. The
actual name of the attribute is defined by the specific analysis
class. The format and content is also dependent on the analysis; for
example, the RMSD analysis class :class:`pmda.rms.RMSD` produces a
numpy array of the RMSD time series.


.. _example-parallel-rmsd:

Example: Parallel RMSD
======================
	    
In order to use the parallel RMSD calculator :class:`pmda.rms.RMSD`,
import the module, set up two
:class:`~MDAnalysis.core.groups.AtomGroup` instances, instantiate the
class and directly use its :meth:`~pmda.rms.RMSD.run` method:

.. code:: python

   import MDAnalysis as mda
   from pmda import rms

   u = mda.Universe(top, traj)
   ref = mda.Universe(top, traj)

   rmsd_ana = rms.RMSD(u.atoms, ref.atoms).run(n_jobs=-1)

   print(rmsd_ana.rmsd)

The resulting ``rmsd_ana`` instance contains the array of the RMSD
time series in the attribute :attr:`~pmda.rms.RMSD.rmsd`.



.. _selection:
   https://www.mdanalysis.org/docs/documentation_pages/selections.html
