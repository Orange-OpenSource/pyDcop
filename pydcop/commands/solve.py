# BSD-3-Clause License
#
# Copyright 2017 Orange
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
.. _pydcop_commands_solve:

pydcop solve
============

``pydcop solve`` solves a static DCOP by running locally a set of
agents.

Synopsis
--------

::

  pydcop solve --algo <algo> [--algo_params <params>]
               [--distribution <distribution>]
               [--mode <mode>]
               [--collect_on <collect_mode>]
               [--period <p>]
               [--run_metrics <file>]
               [--end_metrics <file>]
               [--delay <delay>]
               [--uiport <port>]
               <dcop_files>


Description
-----------

The ``solve`` command is a shorthand for the
:ref:`agent<pydcop_commands_agent>` and
:ref:`orchestrator<pydcop_commands_orchestrator>`
commands.
It solves a DCOP on the local machine, automatically creating all the agents
as specified in the dcop definitions and the orchestrator
(required for bootstrapping and collecting metrics and results).

When using ``solve`` all agents run on the same machine.  Depending on the
``--mode`` parameter, agents will be created as threads (lightweight)
or as process (heavier, but better parallelism on a multi-core cpu).
Using ``--mode thread`` (the default) agents communicate in memory (without
network), which scales easily to more than 100 agents.


Notes
------
Depending on the DCOP algorithm, the solve process may or may not stop
automatically.
For example, :ref:`DPOP<implementation_reference_algorithms_dpop>`
has a clear termination condition and the command will return once this
condition is reached.
On the other hand, some other algorithm like
:ref:`MaxSum<implementation_reference_algorithms_maxsum>` have
no clear termination condition.

Some algorithm have an optional termination condition, which can be passed
as an argument to the algorithm.
With :ref:`MGM<implementation_reference_algorithms_mgm>` for example,
you can use ``--algo_params stop_cycle:<cycle_count>`` ::

  pydcop solve --algo mgm --algo_params stop_cycle:20 \\
               --collect_on cycle_change --run_metric ./metrics.csv \\
               graph_coloring_50.yaml

For algorithms with no termination condition, you should use the global
``--timeout`` option.
Note that the ``--timeout`` is used as a timeout for the solve process only.
Bootstrapping the system and gathering metrics take additional time,
which is not accounted for in the timeout.
This means that the solve command may take more time to return
than the time set with the global ``--timeout`` option.

You can always stop the process manually with ``CTRL+C``.
Here again, the system may take a few seconds to stop.

See Also
--------

**Commands:** :ref:`pydcop_commands_agent`, :ref:`pydcop_commands_orchestrator`

**Tutorials:** :ref:`tutorials_analysing_results` and
:ref:`tutorials_deploying_on_machines`


Output
------

This commands outputs the end results of the solve process.
A detailed description of this output is described in the
:ref:`tutorials_analysing_results` tutorial.


Options
-------

``--algo <dcop_algorithm>`` / ``-a <dcop_algorithm>``
  Name of the dcop algorithm, e.g. 'maxsum', 'dpop', 'dsa', etc.

``--algo_params <params>`` / ``-p <params>``
  Optional parameter for the DCOP algorithm, given as string
  ``name:value``.
  This option may be used multiple times to set several parameters.
  Available parameters depend on the algorithm,
  check :ref:`algorithms documentation<implementation_reference_algorithms>`.

``--distribution <distribution>`` / ``-d <distribution>``
  Either a :ref:`distribution algorithm<implementation_reference_distributions>`
  (``oneagent``, ``adhoc``, ``ilp_fgdp``, etc.) or
  the path to a yaml file containing the distribution.
  (see :ref:`yaml format<usage_file_formats_distribution>`.)
  If not given, ``oneagent`` is used.

``--mode <mode>`` / ``-m``
    Indicated if agents must be run as threads (default) or processes.
    either ``thread`` or ``process``

``--collect_on <collect_mode>`` / ``-c``
    Metric collection mode, one of ``value_change``, ``cycle_change``,
    ``period``.
    See :ref:`tutorials_analysing_results` for details.

``--period <p>``
    When using ``--collect_on period``, the period in second for metrics
    collection.
    See :ref:`tutorials_analysing_results` for details.

``--run_metrics <file>``
    Path to a file or file name. Run-time metrics will be written to that file
    (csv format). If the value is a path, the directory will be created if it does
    not exist. Otherwise the file will be created in the current directory.

``--end_metrics <file>``
    Path to a file or file name. Result's metrics will be appended to that file
    (csv format). If the value is a path, the directory will be created if it does not
    exist. Otherwise the file will be created in the current directory.

``--delay <delay>``
  An optional delay between message delivery, in second. This delay
  only applies to algorithm's messages and is useful when you want to
  observe (for example with the GUI) the behavior of the algorithm at
  runtime.

``--uiport``
  The port on which the ui-server will be listening.
  This port is used for the orchestrator and incremented for each following
  agent. If not given, no ui-server will be started for any agent.


``<dcop_files>``
  One or several paths to the files containing the dcop. If several paths are
  given, their content is concatenated as used a the
  :ref:`yaml definition<usage_file_formats_dcop>` for the
  DCOP.


Examples
--------

The simplest form is to simply specify an algorithm and a dcop yaml file.
Beware that, depending on the algorithm, this command may never return and
need to be stopped with CTRL+C::

    pydcop solve --algo maxsum  graph_coloring1.yaml
    pydcop -t 5 solve --algo maxsum  graph_coloring1.yaml


Solving with MGM, with two algorithm parameter and a log configuration file::

  pydcop --log log.conf solve --algo mgm --algo_params stop_cycle:20 \\
                              --algo_params break_mode:random  \\
                              graph_coloring.yaml \\

"""
import csv
import json
import logging
import os
import multiprocessing
import sys
import threading
import traceback
from functools import partial
from queue import Queue, Empty
from threading import Thread

from pydcop.algorithms import list_available_algorithms
from pydcop.commands._utils import build_algo_def, _error, _load_modules
from pydcop.dcop.yamldcop import load_dcop_from_file
from pydcop.distribution.yamlformat import load_dist_from_file
from pydcop.infrastructure.run import run_local_thread_dcop, run_local_process_dcop


logger = logging.getLogger("pydcop.cli.solve")


def set_parser(subparsers):

    algorithms = list_available_algorithms()
    logger.debug("Available DCOP algorithms %s", algorithms)

    parser = subparsers.add_parser("solve", help="solve static dcop")
    parser.set_defaults(func=run_cmd)
    parser.set_defaults(on_timeout=on_timeout)
    parser.set_defaults(on_force_exit=on_force_exit)

    parser.add_argument(
        "dcop_files",
        type=str,
        nargs="+",
        help="The DCOP, in one or several yaml file(s)",
    )

    parser.add_argument(
        "-a",
        "--algo",
        choices=algorithms,
        required=True,
        help="The algorithm for solving the dcop",
    )
    parser.add_argument(
        "-p",
        "--algo_params",
        type=str,
        action="append",
        help="Optional parameters for the algorithm, given as "
        "name:value. Use this option several times "
        "to set several parameters.",
    )

    parser.add_argument(
        "-d",
        "--distribution",
        type=str,
        default="oneagent",
        help="A yaml file with the distribution or algorithm "
        "for distributing the computation graph, if not "
        "given the `oneagent` will be used (one "
        "computation for each agent)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="thread",
        choices=["thread", "process"],
        help="run agents as threads or processes",
    )

    parser.add_argument(
        "-c",
        "--collect_on",
        choices=["value_change", "cycle_change", "period"],
        default=None,
        help='When should a "new" assignment be observed',
    )

    parser.add_argument(
        "--period",
        type=float,
        default=None,
        help="Period for collecting metrics. only available "
        "when using --collect_on period. Defaults to 1 "
        "second if not specified",
    )

    parser.add_argument(
        "--run_metrics",
        type=str,
        default=None,
        help="Path to a file or file name. Run-time metrics will "
        "be written to that file (csv format). If the value is a "
        "path, the directory will be created if it does not exist. "
        "Otherwise the file will be created in the current directory.",
    )

    parser.add_argument(
        "--end_metrics",
        type=str,
        default=None,
        help="Path to a file or file name. Result's metrics will "
        "be appended to that file (csv format). If the value is a "
        "path, the directory will be created if it does not exist. "
        "Otherwise the file will be created in the current directory.",
    )

    parser.add_argument(
        "--infinity",
        "-i",
        default=float("inf"),
        type=float,
        help="Argument to determine the value used for "
        "infinity in case of hard constraints, "
        "for algorithms that do not use symbolic "
        "infinity. Defaults to 10 000",
    )

    parser.add_argument(
        "--delay",
        default=None,
        type=float,
        help="an optional delay between message delivery, "
        " in second. This delay only applies to "
        "algorithm's messages and is useful when you "
        "want to observe (for example with the UI) the "
        "behavior of the algorithm at runtime",
    )

    parser.add_argument(
        "--uiport",
        type=int,
        default=None,
        help="The port on which the ui-server will be "
        "listening. This port is used for the orchestrator"
        "and incremented for each following agent. If not "
        "given, no ui-server will be started for any "
        "agent.",
    )

DISTRIBUTION_METHODS = ["oneagent", "adhoc", "ilp_fgdp", "heur_comhost", "oilp_secp_fgdp", "gh_secp_fgdp", "gh_secp_cgdp", "oilp_cgdp", "gh_cgdp"]


dcop = None
orchestrator = None
INFINITY = None

# Files for logging metrics
columns = {
    "cycle_change": [
        "cycle",
        "time",
        "cost",
        "violation",
        "msg_count",
        "msg_size",
        "status",
    ],
    "value_change": [
        "time",
        "cycle",
        "cost",
        "violation",
        "msg_count",
        "msg_size",
        "status",
    ],
    "period": ["time", "cycle", "cost", "violation", "msg_count", "msg_size", "status"],
}

collect_on = None
run_metrics = None
end_metrics = None

timeout_stopped = False
output_file = None


def add_csvline(file, mode, metrics):
    data = [metrics[c] for c in columns[mode]]
    with open(file, mode="at", encoding="utf-8", newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(data)


def collect_tread(collect_queue: Queue, csv_cb):
    while True:
        try:
            t, metrics = collect_queue.get()

            if csv_cb is not None:
                csv_cb(metrics)

        except Empty:
            pass
        # FIXME : end of run ?


def prepare_metrics_files(run, end, mode):
    """
    Prepare files for storing metrics, if requested.
    Returns a cb that can be used to log metrics in the run_metrics file.
    """
    global run_metrics, end_metrics
    if run is not None:
        run_metrics = run
        # delete run_metrics file if it exists, create intermediate
        # directory if needed
        if os.path.exists(run_metrics):
            os.remove(run_metrics)
        else:
            f_dir = os.path.dirname(run_metrics)
            if f_dir and not os.path.exists(f_dir):
                os.makedirs(f_dir)
        # Add column labels in file:
        with open(run_metrics, "w", encoding="utf-8", newline="") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(columns[mode])
        csv_cb = partial(add_csvline, run_metrics, mode)
    else:
        csv_cb = None

    if end is not None:
        end_metrics = end
        e_dir = os.path.dirname(end_metrics)
        if e_dir and not os.path.exists(e_dir):
            os.makedirs(e_dir)
        # Add column labels in file:
        if not os.path.exists(end_metrics):
            with open(end_metrics, "w", encoding="utf-8", newline="") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(columns[mode])

    return csv_cb


def run_cmd(args, timer=None, timeout=None):
    logger.debug('dcop command "solve" with arguments {}'.format(args))

    global INFINITY, collect_on, output_file
    INFINITY = args.infinity
    output_file = args.output
    collect_on = args.collect_on

    period = None
    if args.collect_on == "period":
        period = 1 if args.period is None else args.period
    else:
        if args.period is not None:
            _error('Cannot use "period" argument when collect_on is not ' '"period"')

    csv_cb = prepare_metrics_files(args.run_metrics, args.end_metrics, collect_on)

    if args.distribution in DISTRIBUTION_METHODS:
        dist_module, algo_module, graph_module = _load_modules(
            args.distribution, args.algo
        )
    else:
        dist_module, algo_module, graph_module = _load_modules(None, args.algo)

    global dcop
    logger.info("loading dcop from {}".format(args.dcop_files))
    dcop = load_dcop_from_file(args.dcop_files)
    logger.debug(f"dcop  {dcop} ")

    # Build factor-graph computation graph
    logger.info("Building computation graph ")
    cg = graph_module.build_computation_graph(dcop)
    logger.debug("Computation graph: %s ", cg)

    logger.info("Distributing computation graph ")
    if dist_module is not None:

        if not hasattr(algo_module, "computation_memory"):
            algo_module.computation_memory = lambda *v, **k: 0
        if not hasattr(algo_module, "communication_load"):
            algo_module.communication_load = lambda *v, **k: 0

        distribution = dist_module.distribute(
            cg,
            dcop.agents.values(),
            hints=dcop.dist_hints,
            computation_memory=algo_module.computation_memory,
            communication_load=algo_module.communication_load,
        )
    else:
        distribution = load_dist_from_file(args.distribution)
    logger.debug("Distribution Computation graph: %s ", distribution)

    logger.info("Dcop distribution : {}".format(distribution))

    algo = build_algo_def(algo_module, args.algo, dcop.objective, args.algo_params)

    # Setup metrics collection
    collector_queue = Queue()
    collect_t = Thread(
        target=collect_tread, args=[collector_queue, csv_cb], daemon=True
    )
    collect_t.start()

    global orchestrator
    if args.mode == "thread":
        orchestrator = run_local_thread_dcop(
            algo,
            cg,
            distribution,
            dcop,
            INFINITY,
            collector=collector_queue,
            collect_moment=args.collect_on,
            period=period,
            delay=args.delay,
            uiport=args.uiport,
        )
    elif args.mode == "process":

        # Disable logs from agents, they are in other processes anyway
        agt_logs = logging.getLogger("pydcop.agent")
        agt_logs.disabled = True

        # When using the (default) 'fork' start method, http servers on agent's
        # processes do not work (why ?)
        multiprocessing.set_start_method("spawn")
        orchestrator = run_local_process_dcop(
            algo,
            cg,
            distribution,
            dcop,
            INFINITY,
            collector=collector_queue,
            collect_moment=args.collect_on,
            period=period,
            delay=args.delay,
            uiport=args.uiport,
        )
    try:
        orchestrator.deploy_computations()
        orchestrator.run(timeout=timeout)
        if timer:
            timer.cancel()
        if not timeout_stopped:
            if orchestrator.status == "TIMEOUT":
                _results("TIMEOUT")
                sys.exit(0)
            elif orchestrator.status != "STOPPED":
                _results("FINISHED")
                sys.exit(0)

        # in case it did not stop, dump remaining threads

    except Exception as e:
        logger.error(e, exc_info=1)
        orchestrator.stop_agents(5)
        orchestrator.stop()
        _results("ERROR")


def on_timeout():
    logger.debug("cli timeout ")
    # Timeout should have been handled by the orchestrator, if the cli timeout
    # has been reached, something is probably wrong : dump threads.
    for th in threading.enumerate():
        print(th)
        traceback.print_stack(sys._current_frames()[th.ident])
        print()

    if orchestrator is None:
        logger.debug("cli timeout with no orchestrator ?")
        return
    global timeout_stopped
    timeout_stopped = True
    # Stopping agents can be rather long, we need a big timeout !
    logger.debug("stop agent on cli timeout ")
    orchestrator.stop_agents(20)
    logger.debug("stop orchestrator on cli timeout ")
    orchestrator.stop()
    _results("TIMEOUT")
    # sys.exit(0)
    os._exit(2)


def on_force_exit(sig, frame):
    if orchestrator is None:
        return
    orchestrator.status = "STOPPED"
    orchestrator.stop_agents(5)
    orchestrator.stop()
    _results("STOPPED")
    os._exit(2)


import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def _results(status):
    """
    Outputs results and metrics on stdout and trace last metrics in csv
    files if requested.

    :param status:
    :return:
    """

    metrics = orchestrator.end_metrics()
    metrics["status"] = status
    global end_metrics, run_metrics
    if end_metrics is not None:
        add_csvline(end_metrics, collect_on, metrics)
    if run_metrics is not None:
        add_csvline(run_metrics, collect_on, metrics)

    if output_file:
        with open(output_file, encoding="utf-8", mode="w") as fo:
            fo.write(json.dumps(metrics, sort_keys=True, indent="  ", cls=NumpyEncoder))

    print(json.dumps(metrics, sort_keys=True, indent="  ", cls=NumpyEncoder))
