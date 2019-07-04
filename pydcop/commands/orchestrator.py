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
.. _pydcop_commands_orchestrator:

pydcop orchestrator
===================

``pydcop orchestrator`` runs an orchestrator.

Synopsis
--------

::

  pydcop orchestrator --algo <algo> [--algo_params <params>]
                      --distribution <distribution>
                      [--address <ip_addr>] [--port <port>]
                      [--uiport <uiport>]
                      [--collect_on <collect_mode>] [--period <p>]
                      [--run_metrics <file>]
                      [--end_metrics <file>]
                      <dcop_files>


Description
-----------

Runs an orchestrator, which waits for agents, deploys on them the computations
required to solve the DCOP with the requested algorithm and collects
selected values from agents. Agents must be run separately using the
``agent`` command (see. :ref:`pydcop_commands_agent`).

The ``orchestrator`` command support the global ``--timeout`` argument and can
also be stopped using ``CTRL+C``.

When the orchestrator stops, it request all agents to stop and displays the
current DCOP solution (with associated cost) in yaml.

See Also
--------

**Commands:** :ref:`pydcop_commands_agent`, :ref:`pydcop_commands_solve`

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
  If not given, ``oneagent`` is used.

``--collect_on <collect_mode>`` / ``-c``
    Metric collection mode, one of ``value_change``, ``cycle_change``,
    ``period``.
    See :ref:`tutorials_analysing_results` for details.

``--period <p>``
    When using ``--collect_on period``, the period in second for metrics
    collection.
    See :ref:`tutorials_analysing_results` for details.

``--run_metrics <file>``
    File to store store metrics.
    See :ref:`tutorials_analysing_results` for details.

``--end_metrics <file>``
    End metrics (i.e. when the solve process stops) will be appended to this
    file (in csv).

``--address <ip_address>``
  Optional IP address the orchestrator will listen on.
  If not given we try to use the primary IP address.

``--port <port>``
  Optional port the orchestrator will listen on.
  If not given we try to use port 9000.

``--uiport <port>``
  Optional port the orchestrator's ui-server (only needed when using the GUI).
  If not given no ui-server is started.

``<dcop_files>``
  One or several paths to the files containing the dcop. If several paths are
  given, their content is concatenated as used a the
  :ref:`yaml definition<usage_file_formats_dcop>` for the
  DCOP.

Examples
--------

Running an orchestrator for 5 seconds (on default IP and port),
to solve a graph coloring DCOP with ``maxsum``.
Computations are distributed
using the  ``adhoc`` algorithm::

  pydcop --timeout 5 orchestrator -a maxsum -d adhoc graph_coloring.yaml

Running an orchestrator that collects metrics every 0.2 second and run
the :ref:`MGM algorithm<implementation_reference_algorithms_mgm>`
on agents for 20 cycles::

  pydcop -v 3 orchestrator --algo mgm --algo_param stop_cycle:20 \\
                           --collect_on period --period 0.2 \\
                           --run_metrics ./orch_metrics_period.csv \\
                           --address 192.168.1.2 --port 10000 \\
                           graph_coloring_3agts.yaml
"""

import json
import logging
import os
import sys
import threading
import traceback
from functools import partial
from queue import Queue, Empty
from threading import Thread

import multiprocessing
from importlib import import_module
from time import time

from pydcop.algorithms import list_available_algorithms, load_algorithm_module
from pydcop.commands._utils import build_algo_def
from pydcop.dcop.yamldcop import load_dcop_from_file, load_scenario_from_file
from pydcop.distribution.yamlformat import load_dist_from_file
from pydcop.infrastructure.communication import HttpCommunicationLayer
from pydcop.infrastructure.orchestrator import Orchestrator

logger = logging.getLogger("pydcop.cli.orchestrator")


def set_parser(subparsers):
    algorithms = list_available_algorithms()
    logger.debug("Available DCOP algorithms %s", algorithms)

    parser = subparsers.add_parser("orchestrator", help="run a standalone orchestrator")
    parser.set_defaults(func=run_cmd)
    parser.set_defaults(on_timeout=on_timeout)
    parser.set_defaults(on_force_exit=on_force_exit)

    parser.add_argument("dcop_files", type=str, nargs="+", help="dcop file(s)")
    parser.add_argument(
        "-a", "--algo", choices=algorithms, help="algorithm for solving the dcop"
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
        default="oneagent",
        help="A yaml file with the distribution or algorithm "
        "for distributing the computation graph, if not "
        "given the `oneagent` will be used (one "
        "computation for each agent)",
    )

    parser.add_argument(
        "-c",
        "--collect_on",
        choices=["value_change", "cycle_change", "period"],
        default="value_change",
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
        help="Use this option to regularly store the data " "in a csv file.",
    )

    parser.add_argument(
        "--end_metrics",
        type=str,
        default=None,
        help="Use this option to append the metrics of the "
        "end of the run to a csv file.",
    )

    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help="IP address the orchestrator will listen on. If "
        "not given we try to use the primary IP address.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port the orchestrator will listen on. If "
        "not given we try to use port 9000.",
    )

    parser.add_argument(
        "--uiport",
        type=int,
        default=None,
        help="The port on which the ui-server will be listening"
        ". If not given, no ui-server will "
        "be started for this orchestrator.",
    )

    parser.add_argument(
        "--ktarget",
        type=int,
        default=None,
        help="The target resiliency level. If not given, computations are "
        "not replicated and the system is not resilient.",
    )

    parser.add_argument(
        "-s",
        "--scenario",
        required=False,
        default=None,
        help="scenario file. When using a scenario, replication is automatically activated",
    )


DISTRIBUTION_METHODS = ["oneagent", "adhoc", "ilp_fgdp", "heur_comhost", "oilp_secp_fgdp", "gh_secp_fgdp", "gh_secp_cgdp", "oilp_cgdp", "gh_cgdp"]

orchestrator = None
start_time = 0

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
    line = ",".join([str(d) for d in data])

    with open(file, mode="at", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


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
        headers = ",".join(columns[mode])
        with open(run_metrics, "w", encoding="utf-8") as f:
            f.write(headers)
            f.write("\n")
        csv_cb = partial(add_csvline, run_metrics, mode)
    else:
        csv_cb = None

    if end is not None:
        end_metrics = end
        if not os.path.exists(os.path.dirname(end_metrics)):
            os.makedirs(os.path.dirname(end_metrics))
        # Add column labels in file:
        if not os.path.exists(end_metrics):
            headers = ",".join(columns[mode])
            with open(end_metrics, "w", encoding="utf-8") as f:
                f.write(headers)
                f.write("\n")

    return csv_cb


def run_cmd(args, timer=None, timeout=None):
    logger.debug('dcop command "orchestrator" with arguments {} '.format(args))

    global collect_on, output_file
    output_file = args.output
    collect_on = args.collect_on

    dcop_yaml_files = args.dcop_files

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

    logger.info("loading dcop from {}".format(dcop_yaml_files))
    dcop = load_dcop_from_file(dcop_yaml_files)

    if args.scenario:
        logger.info("loading scenario from {}".format(args.scenario))
        scenario = load_scenario_from_file(args.scenario)
    else:
        logger.debug("No scenario")
        scenario = None

    # Build factor-graph computation graph
    logger.info("Building computation graph for dcop {}".format(dcop_yaml_files))
    cg = graph_module.build_computation_graph(dcop)

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

    logger.info("Dcop distribution : {}".format(distribution))

    algo = build_algo_def(algo_module, args.algo, dcop.objective, args.algo_params)

    # When using the (default) 'fork' start method, http servers on agent's
    # processes did not work (why ?), but seems to be ok now ?!
    # multiprocessing.set_start_method('spawn')

    # FIXME
    infinity = 10000

    # Setup metrics collection
    collector_queue = Queue()
    collect_t = Thread(
        target=collect_tread, args=[collector_queue, csv_cb], daemon=True
    )
    collect_t.start()

    if args.ktarget:
        ktarget = args.ktarget
    else:
        if scenario:
            logger.debug("Scenario without k target, use 3 as default level")
            ktarget = 3

    global orchestrator, start_time
    port = args.port if args.port else 9000
    addr = args.address if args.address else None
    comm = HttpCommunicationLayer((addr, port))
    orchestrator = Orchestrator(
        algo,
        cg,
        distribution,
        comm,
        dcop,
        infinity,
        collector=collector_queue,
        collect_moment=args.collect_on,
        collect_period=period,
        ui_port=args.uiport,
    )

    try:
        start_time = time()
        logger.debug(f"Starting Orchestrator")
        orchestrator.start()
        logger.debug(f"Deploying computations")
        orchestrator.deploy_computations()
        if scenario:
            logger.debug(f"Starting Replication, targert {ktarget}")
            orchestrator.start_replication(ktarget)
            if orchestrator.wait_ready():

                orchestrator.run(scenario=scenario, timeout=timeout)
        else:
            logger.debug("No scenario, run the problem directly")
            orchestrator.run(timeout=timeout)
        if not timeout_stopped:
            if orchestrator.status == "TIMEOUT":
                _results("TIMEOUT")
                sys.exit(0)
            else:
                _results("FINISHED")
                sys.exit(0)

    except Exception as e:
        logger.error(e, exc_info=1)
        orchestrator.stop_agents(5)
        orchestrator.stop()
        _results("ERROR")


def on_force_exit(sig, frame):
    print("FORCE EXIT")
    # Avoid blocking if stopping when all agents have not registered yet
    if not orchestrator.mgt.all_registered.is_set():
        orchestrator.mgt.all_registered.set()
    orchestrator.stop_agents(10)
    orchestrator.stop()
    _results("STOPPED")
    sys.exit(0)


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

    orchestrator.stop_agents(20)
    orchestrator.stop()
    _results("TIMEOUT")
    sys.exit(0)


def _load_modules(dist, algo):
    dist_module, algo_module, graph_module = None, None, None
    if dist:
        try:
            dist_module = import_module("pydcop.distribution.{}".format(dist))
            # TODO check the imported module has the right methods ?
        except ImportError:
            _error("Could not find distribution method {}".format(dist))

    try:
        algo_module = load_algorithm_module(algo)
        # TODO check the imported module has the right methods ?

        graph_module = import_module(
            "pydcop.computations_graph.{}".format(algo_module.GRAPH_TYPE)
        )
    except ImportError:
        _error(
            "Could not find computation graph type: {}".format(algo_module.GRAPH_TYPE)
        )

    return dist_module, algo_module, graph_module


def _error(msg):
    print("Error: {}".format(msg))
    sys.exit(2)


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
