

.. _concepts_resilient_dcop:

Resilient DCOP
==============

In most articles on DCOPs,
variables and computation are mapped to a static set of agents,
which is supposed to persist over the resolution time of the system.

However, as for most distributed systems,
DCOP are often used on problems where the environment is dynamic in nature.
Agents may enter of leave the system at any time
and the system needs to adapt to these changes.

This is typically the case when applying DCOP on real problems
like Internet-of-Things (IoT), Ambient intelligence, sensor networks
or smart-grid,
where computations run on distributed, highly heterogeneous,
nodes and where a central coordination might not desirable or even not possible.

In such settings, systems must be able to
cope with node additions and failures:
when a node stops responding,
other nodes in the system must take responsibility and
run the orphaned computations.
Similarly, when a new node is added in the system,
it might be useful to reconsider the distribution of computations in order to
take advantage of the newcomer’s computational capabilities,
as proposed in :cite:`rust_deployment_2017`.

Here we use the term **computation**,
which represents the basic unit of work needed when solving a DCOP,
and generally maps to variables (and factors for some DCOP algorithms)
in the DCOP. See the page on :ref:`concepts_graph` for more details.

pyDCOP introduces a notion of a resilient DCOP and proposes
mechanisms to cope with these changes in the infrastructure.
This mechanisms are described in more details in
:cite:`rust_self-organized_2018`.



Initial distribution
--------------------

As explained in :ref:`concepts_distribution`,
the computations of the DCOP need to be distributed on the agent / devices
at startup.
In many cases, this initial distribution can computed centrally
as part of a bootstrapping process.


Computation Replication
-----------------------

One pre-requisite to resilience is
to still have access to the definition of every computation after a failure.
One approach is to keep replicas (copies of definitions) of each
computation on different agents.
Provided that k replicas are placed on k different agents,
no matter the subset of up to k agents that fails there will always be at least
one replica left after the failure.
This approach is classically found in distributed database systems

The ideal placement of these replicas is far from trivial
and is an optimization problem in itself, which depends strongly on the
nature of the real problem solved by the DCOP.
See :cite:`rust_self-organized_2018` for an example.

pyDCOP currently proposes one distributed replication algorithm,
called **DRPM**,
which is a distributed version of iterative lengthening
(uniform cost search based on path costs).


Distribution reparation
-----------------------

Given a mechanism to replicate computations,
the DCOP distribution can be repaired
when an agents fails or is removed from the system.
This reparation process must decide which agent
will host the orphaned computations
(aka the computations that were hosted on the departed agent)
and should ideally be decentralized.

Here again, several approach can be designed to handle such reparation,
like those presented in :cite:`rust_deployment_2017` and
:cite:`rust_self-organized_2018`, which are currently implemented in pyDCOP.

