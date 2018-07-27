

.. _usage_cli_ref:

pyDCOP command line reference
=============================


.. toctree::
   :maxdepth: 1

   cli/agent
   cli/orchestrator
   cli/run
   cli/solve
   cli/graph
   cli/replica_dist
   cli/generate
   cli/distribute


pyDCOP command line interface can be called either using ``dcop_cli.py`` or
``pydcop`` (which is simply a shell script pointing to ``dcop_cli.py``). For example,
the two following commands are strictly equivalent::

  dcop_cli.py --version
  pydcop --version

pyDCOP command line script works with an 'command' concept, similar to ``git``
(e.g. in ``git commit``, ``commit`` is a 'command' for the ``git`` cli). Each
action defines its own arguments, which must be given after the command name
and are documented in their respective page.
Additionally, some options apply to many commands and must be given
**before** the command, for example in the following ``-t`` and ``-v`` are
**global options** and ``--algo`` is an option of the ``solve`` command::

  pydcop -t 5 -v 3 solve --algo maxsum  graph_coloring.yaml

pydcop supports the following global options::

  pydcop [--version] [--timeout <timeout>] [--verbosity <level>]
         [--log <log_conf_file>]


.. _usage_cli_ref_options:

``--version``
  Outputs pydcop version and exits.

``--timeout <timeout>`` / ``-t <timeout>``
  Set a global timeout (in seconds) for the command.

``--output <output_file>``
  Write the command's output to a file instead of std out.

``-verbose <level>`` / ``-v <level>``
  Set verbosity level (0-3). Defaults to level 0, which should be used when you
  need a parsable output as it only logs errors and the output is only the yaml
  formatted result of the command.

``--log <long_conf_file>``
  Log configuration file. Can be used instead of ``-verbose`` for precise
  control over log (filtering, output to several files, etc.). This
  file uses the `standard python log configuration file format <https://docs
  .python.org/3/library/logging.config.html#configuration-file-format>`_ .
  The following sample file can be used as a starting point to build your own
  custom log configuration : :download:`log.conf<cli/log.conf>`.

Additionally the ``--help`` / ``-h`` option can always be used both as a
global option and as a command option.
Calling ``pydcop --help`` outputs a general help for pyDCOP command line
interface, with a list of available commands.
``pydcop <command> --help`` outputs help for this specific command.



