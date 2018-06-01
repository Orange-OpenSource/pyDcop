
.. _tutorials_analysing_results:


Analysing results
=================

End results
-----------

In the :ref:`first tutorial<tutorials_getting_started>` you solved a very
simple DCOP using the following command::

  pydcop solve --algo dpop graph_coloring.yaml

This command outputs many results, given in json so that you can parse then
easily, for example when writing more complex scripts for
experimentation.
You can also get these results directly in a file using the ``--output`` global
option (see. :ref:`pyDcop cli dcoumentation<usage_cli_ref_options>`):
``pydcop --output results.json solve --algo dpop graph_coloring.yaml``

You may have noticed that, even though the DCOP has only 3 variables and a
handful of constraints, the results is quite long and contains a lot of
information.

At the top level you can find global results, which apply to the DCOP as a
whole::

    {
      "agt_metrics": {
        // metrics for each agent, see bellow
      },
      "assignment": {
        "v1": "R",
        "v2": "G",
        "v3": "R"
      },
      "cost": -0.1,
      "cycle": 1,
      "msg_count": 29,
      "msg_size": 8,
      "status": "FINISHED",
      "time": 0.006329163996269926,
      "violation": 0
    }

* ``status``: indicates how the command was stopped, can be one of

  * ``FINISHED``, when all computation finished,
  * ``TIMEOUT``, when the command was interrupted because it reached timeout
    set with the ``--timeout`` option,
  * ``STOPPED``, when the command was interrupted with ``CTRL+C``.

* ``assignement``: contains the assignment that was found by the DCOP
  algorithm.
* ``cost``: the cost for this assignment (i.e. the sum of the cost of all
  constraints in the DCOP).
* ``violation``: the number of violated hard constraints, when using DCOP with a
  mix of soft and hard constraints (hard constraints support is still in
  experimental stage and not documented).
* ``msg_size``: the total size of messages exchanged between agents.
* ``msg_count``: the number of messages exchanged between agents.
* ``time``: the elapsed time, in second
* ``cycles``, the number of cycles for  DCOP computations.
  In this example it is always 0 has DPOP has no notion of cycle.


The ``agt_metrics`` section contains one entry for each of the agents in the
DCOP. Each of these entries contains several items::

    "a1": {
      "activity_ratio": 0.24987247830102727,
      "count_ext_msg": {
        "v2": 2
      },
      "cycles": {
        "v2": 0
      },
      "size_ext_msg": {
        "v2": 4
      }
    },


* ``count_ext_msg`` and ``size_ext_msg``, which contain count and size of
  messages sent and received by each DCOP computations hosted on this agent
* ``cycles``, the number of cycles for each DCOP computation hosted on this
  agent.
  In this example it is always 0 has DPOP has no notion of cycle.
* ``activity_ratio``, the ratio of *active time* on total elapsed time,
   where active time is defined as the time the agent spent handling a
   message (as opposed to waiting for messages).


Logs
----

By default, the :ref:`solve<pydcop_commands_solve>` command (like all other
pyDCOP commands) only gives you the end metrics. You can enable logs by
adding the ``-v`` option with the requested level::

  pydcop -v 2 solve --algo dpop graph_coloring.yaml

Level 1 displays only warnings messages, level 2 displays warnings and info
messages and level 3 all messages (and can be quite verbose! )

For more control over logs, you can use the ``--log`` option.


TODO: with conf file


Run-time metrics
----------------

The output of the :ref:`solve<pydcop_commands_solve>` command only gives you the
end results of the command, but sometime you need to know what happens during
the execution of the DCOP algorithm.



Plotting the results
--------------------

matplot lib example.
