.. _implementation_agents:


Agents
======


Agents can be responsible for several variables
and can host one or several computations.
Each agent has it's own thread and message queue
and manage communication with other agents.
An agent delivers messages to the computations it hosts
and send messages on their behalf.

When running a dcop, all agents
can run in the same process or in different processes.
In the first case communication is implemented
with direct object exchanges between the agent's threads.
In the second case,
agents may run on different computers and  communication uses the network.
Current implementation is based on HTTP+JSON
but other network communication mecanism (zeromq, CoAP, BSON, etc.)
could easily be implemented by subclassing
:class:`.CommunicationLayer`


The Orchestrator is a special agent that is not part of the DCOP :
it's role is to bootstrap the solving process
by distributing the computations on the agents.
It also collects metrics for benchmark purpose.
Once the system is started (and if no metric is collected),
the orchestrator could be removed.


Agents objects are implemented in 3 different modules:

* pydcop.dcop.infrastructure.agents
* pydcop.dcop.infrastructure.orchestratedagents
* pydcop.dcop.infrastructure.orchestrator


.. currentmodule:: pydcop.infrastructure

.. autosummary::
   :toctree: pydcop.infrastructure

   agents
   orchestratedagents
   orchestrator


