

.. _implementation_reference_distributions:

pydcop.distribution
===================

A distribution method is used to decide which agent hosts each computation.


Distribution methods are implemented in ``pydcop.distribution``.
``object.py`` defines objects that are used by all distribution methods
(``Distribution` and `DistributionHints``).
A distribution method computes the allocation
of a set computations to a set of agents.

List of distribution methods currently implemented in pyDCOP:

.. toctree::
  :maxdepth: 1

  distributions/oneagent
  distributions/ilp_fgdp
  distributions/ilp_compref
  distributions/heur_comhost


Implementing a distribution method
----------------------------------

To implement a new distribution method, one must:

  * create a new module in ``pydcop.distribution``, named after the distribution method
  * define the following methods in this file:
    * ``distribute``
    * ``distribute_remove``
    * ``distribute_add``
