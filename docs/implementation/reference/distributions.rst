

.. _implementation_reference_distributions:

pydcop.distribution
===================

A **distribution** method is used to decide which agent hosts each computation.


Distribution methods are implemented in ``pydcop.distribution``.
``object.py`` defines objects that are used by all distribution methods
(``Distribution` and `DistributionHints``).
A distribution method computes the allocation
of a set computations to a set of agents.

See :ref:`concepts_distribution` for more details on the distribution concept.

pyDCOP currently provides the following distribution methods :

.. toctree::
  :maxdepth: 1

  distributions/oneagent
  distributions/ilp_fgdp
  distributions/ilp_compref
  distributions/heur_comhost


Implementing a distribution method
----------------------------------

To implement a new distribution method, one must:

  * create a new module in ``pydcop.distribution``,
    named after the distribution method
  * define the following methods in this file:

    * ``distribute``, which returns a Distribution object
    * ``distribution_cost``, which return the cost of a distribution. If your
      distribution algorithm does define a notion of cost, you can simply
      return 0 (like the ``oneagent`` distribution)

  * additionally, for dynamic distribution, you can also provide these methods:

    * ``distribute_remove``
    * ``distribute_add``


Base classes
------------

.. autoclass:: pydcop.distribution.oneagent.Distribution
  :members:

