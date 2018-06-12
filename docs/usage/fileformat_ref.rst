
.. _usage_file_formats:

pyDCOP file formats
===================

.. _usage_file_formats_dcop:

DCOP file format
----------------

A DCOP definition is written in yaml-formatted files and must contains
definitions for:

* the objective (min or max)
* variables , and their domains
* constraints, which can be given as intentional or extensional constraints
* agents

It can also contains the following optional elements:

* agents capacity, which is used by some
  :ref:`computations distribution<concepts_distribution>`, mechanisms.
* route and hosting costs, also used for
  :ref:`computations distribution<concepts_distribution>`.


See this sample annotated file for
a quick description of the format:
:download:`dcop_format.yml<file_formats/dcop_format.yml>`.

.. note::
  pyDCOP command line interface accepts DCOP given
  in several files, which will be concatenated before parsing the yaml (see cli
  the command options, for example :ref:`solve<pydcop_commands_solve>`).

  This can be useful, for example when when you want to run the same of
  variables and constraints with different sets of agents ; you can simply
  write a single file for the variables and constraints and one file for each
  set of agents.


.. _usage_file_formats_distribution:

Distribution file format
------------------------

In order to run a DCOP, one must also specify which agent is responsible for
which variable (or, more generally, which agent will host which computation,
see. :ref:`computations distribution<concepts_distribution>`). In pyDCOP, we
call this mapping a **distribution** and it can be described with a yaml file.

Yaml distribution files simply contains a map, where each key is an agent
name and the corresponding value a list of computations hosted on this agent::

  distribution:
    a0: []
    a1: [v1, v2]
    a2: []
    a3: [v2, v3]

See this file for an sample distribution file :
:download:`dist_format.yml<file_formats/dist_format.yml>`.

.. note:: pyDCOP command line interface accepts either a yaml distribution
  file or the name of a distribution algorithm.
  In the latter case, the distribution will be
  automatically generated for the DCOP using the requested method.


.. _usage_file_formats_replication:

Replication file format
-----------------------

When using a resilient DCOP, computations are replicated on several agents.
This replica distribution is generated using the
:ref:`replica_dist<pydcop_commands_replica_dist>` command and can also be
given as a yaml file.

See this file for an replica distribution file :
:download:`replica_dist_format.yml<file_formats/replica_dist_format.yml>`.



.. _usage_file_formats_scenario:

Scenario file format
--------------------

To run a dynamic DCOP, you must also describe the events happening to the
DCOP (agents leaving or entering the system, change of value for an
non-decision variable, etc.). This is done is a scenario file, in yaml.

See this file for an sample scenario file :
:download:`scenario_format.yml<file_formats/scenario_format.yml>`.
