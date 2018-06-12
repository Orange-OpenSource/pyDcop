
.. _tutorials_dynamic_dcops:


Dynamic DCOPs
=============

All the DCOPs we have seen in previous tutorial are *static DCOP* :
the problem definition (variables and constraints)
and the set of agents are supposed to be stable during the solve process.

However, as for most distributed systems,
DCOP are often used on problems where the environment is dynamic in nature.
Agents may enter of leave the system at any time,
the problem itself may evolve *while* it is been worked on,
and the system needs to adapt to these changes.
We call such problems **dynamic DCOP**
(see. :ref:`concepts_resilient_dcop`)

The :ref:`solve<pydcop_commands_solve>` command deals with static DCOP,
the :ref:`run<pydcop_commands_run>` command supports running **dynamic DCOP**.


DCOP and Scenario
-----------------

Running a dynamic DCOP requires injecting events in the system,
otherwise it would be the exact same thing as running a static DCOP
using the :ref:`solve<pydcop_commands_solve>` command.

pyDCOP defines the notion of a scenario, which is an ordered collection of
events that are injected in the system while it is running.
Scenario can be :ref:`written in yaml<usage_file_formats_scenario>`.
Each event in the scenario is either a **delay**,
during which the system runs without any perturbation,
of a collection of **actions**.
For example, in the following scenario,
the agent a008 is removed after a 30 seconds delay::

 events:
   - id: w1
     delay: 30

   - id: e1
     actions:
       - type: remove_agent
         agent: a008


Removing an agent is the only action supported at the moment
(but that should change soon !) .
Other actions could be:

* changing a  constraint value table (the cost of the constraint for a given
  assignment)
* modifying the scope and arity of a constraint
* changing the value of a *read-only* variable

Currently pyDCOP has a partial implementation for these kinds of events,
which are not yet available through the command line interface.


Resiliency and replication
--------------------------

As pyDCOP implementation is currently focused on resilience,
the run command automatically deploy and run a resilient DCOP.

This means that all the computations for the DCOP are automatically
replicated on several agent. Without this step, we would loose computation
when injecting ``remove_agent`` events, and the system would stop working
properly.
The resiliency level and replication method can be selected
when using the :ref:`run<pydcop_commands_run>`  command,
but for now, you don't need to bother with these as sane defaults values are
provided.

For more information on resiliency, replication and reparation, see
:ref:`concepts_resilient_dcop`



Resilient DCOPs
===============
