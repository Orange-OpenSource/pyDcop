
.. _tutorials_deploying_on_machines:


Deploying on several machines
=============================

In the first tutorials, you have been using the
:ref:`solve command<pydcop_commands_solve>` to run your DCOPs.
This command is very convenient as it handles a lot of
*plumbing* details for you,
but it only works if you want to run the whole system on a single machine.
In this tutorial, you will learn how to really distribute your system,
by running different agents on different machines.

Running independent agents
--------------------------

If you want to use several machine to run your DCOP
(remenber, the **D** stands for Distributed !) you need to use the
:ref:`agent<pydcop_commands_agent>`
and :ref:`orchestrator<pydcop_commands_orchestrator>`
commands.

Orchestrator
^^^^^^^^^^^^

The **Orchestrator** is a special agent that is not part of the DCOP:
it's role is to bootstrap the solving process
by distributing the computations on the agents.
It also collects metrics for benchmark purpose.
Once the system is started (and if no metric is collected),
the orchestrator could be removed.
In any case, the orchestrator never participates in the coordination process,
which stays fully decentralised.

The :ref:`orchestrator<pydcop_commands_orchestrator>` command
looks very much like the :ref:`solve command<pydcop_commands_solve>` ;
it takes a DCOP yaml file as input and
supports the same ``--algo``, ``--ditribution``
options.
The main difference is that the orchestrator command only launches an orchestrator,
which then waits for agents to enter the system.
The DCOP algorithm will only be started
once all required agents have been started.

For example, using :download:`this graph coloring problem definition file<graph_coloring_3agts.yaml>`, you can start an orchestrator::

  pydcop -v 3 orchestrator --algo mgm --algo_param stop_cycle:20 \
                           graph_coloring_3agts.yaml

Once the DCOP algorithm finishes, or when reaching the timeout, the
command outputs the end-results.
The content and format is the same than what is described in
:ref:`tutorials_analysing_results`.

All metrics-collection options can also be used with  the
:ref:`orchestrator<pydcop_commands_orchestrator>` and works the same way
than with the :ref:`solve command<pydcop_commands_solve>` command.

Agents
^^^^^^

The :ref:`agent<pydcop_commands_agent>` command launches an agent on the local
machine
(actually it can also launch several agents,
see the :ref:`detailed command documentation<pydcop_commands_agent>`).
Initially, this agent does not know anything about the DCOP (variables,
constraints, etc. ).
It only knows the address of an **orchestrator**,
which is responsible for sending DCOP information
to all agents in the system::

  pydcop -v 3 agent -n a1 -p 9001 --orchestrator 192.168.1.10:9000


Example
^^^^^^^

Instead of using solve, you can run the very simple DCOP used in
:ref:`the first tutorial<tutorials_getting_started>` on different machines.
For easier setup, we reduces the agents number to 3 in this file :
:download:`graph_coloring_3agts.yaml`.


First launch the orchestrator on a machine::

  pydcop -v 3 orchestrator --algo mgm --algo_param stop_cycle:20 \
                           graph_coloring_3agts.yaml

You must check in the logs the ip address and port the orchestrator is
listening on, or you can set it using ``--address`` and ``--port``

Now launch on 3 different machines (or virtual machines) the following
commands to run 3 agents that all use the orchestrator started before
(make sure you give them the right IP address and port!)::

  # Machine 1 runs agent a1
  pydcop -v 3 agent -n a1 -p 9001 --orchestrator 192.168.1.10:9000
  # Machine 2 runs agent a2
  pydcop -v 3 agent -n a2 -p 9001 --orchestrator 192.168.1.10:9000
  # Machine 3 runs agent a3
  pydcop -v 3 agent -n a3 -p 9001 --orchestrator 192.168.1.10:9000

Each agent receives the responsibility for one of the variables from the DCOP
and runs MGM for 20 cycles.
Once each agent has performed 20 cycles, the agents and the orchestrator
commands return.

.. note:: If you know in advance the IP address and port the orchestrator
  will use, you can launch the agents before the orchestrator.
  In that case, agents will periodically attempt to connect to the
  orchestrator, until they can reach it.


Provisioning pyDCOP
-------------------

You may have noticed that the previous section silently assumed that pyDCOP
was installed on every machine you want to use in your system.
Indeed, we use the ``pydcop`` command line application, which is only available
if you have installed pyDCOP!

Of course, you can simply follow the
:ref:`installation instructions<installation>` to install manually pyDCOP on
all your machines, but the process is rather tedious and error prone.
Moreover, if you are working on DCOP algorithms,
you will probably make changes in
pyDCOP implementation (at least in the implementation of your algorithm),
which requires updating it on all your machine, copying the new development
version on all machines, reinstalling it, etc.

When running a large system, one needs to automate this kind of tasks.
To help you with this, we provide as set of ansible playbooks that automates
the installation process. See the :ref:`Provisioning<usage_provisioning>`
guide for full details.
