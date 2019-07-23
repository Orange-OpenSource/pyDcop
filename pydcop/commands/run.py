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
.. _pydcop_commands_run:


pydcop run
==========

Running a (dynamic) DCOP

Synopsis
--------

::

  pydcop run --algo <algo> [--algo_params <params>]
               [--distribution <distribution>]
               [--replication_method <replication method>]
               [--ktarget <resiliency_level>]
               [--mode <mode>]
               [--collect_on <collect_mode>]
               [--period <p>]
               [--run_metrics <file>]
               [--end_metrics <file>]
               --scenario <scenario_file>
               <dcop_files>

Description
-----------
The ``run`` command run a dcop, it is generally used for dynamic dcop where
various events can occur during the life of the system.


Most options are basically the same than the options of the
:ref:`pydcop_commands_solve` command.
The main differences are the optional options for resilent DCOP :
``--ktarget`` and ``--replication method`` and the scenario that contains
events.
See :ref:`usage_file_formats_scenario` for information on the scenario file
format.


When using the ``run``  command, you should use the global
``--timeout`` option.
Note that the ``--timeout`` is used as a timeout for the solve process only.
Bootstrapping the system and gathering metrics take additional time,
which is not accounted for in the timeout.
This means that the run command may take more time to return
than the time set with the global ``--timeout`` option.

You can always stop the process manually with ``CTRL+C``.
Here again, the system may take a few seconds to stop.

See Also
--------

**Commands:** :ref:`pydcop_commands_solve`, :ref:`pydcop_commands_distribute`

**Tutorials:** :ref:`tutorials_analysing_results` and
:ref:`tutorials_dynamic_dcops`


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
  the path to a yaml file containing the distribution
  (see :ref:`yaml format<usage_file_formats_distribution>`).
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
    File to store store metrics.
    See :ref:`tutorials_analysing_results` for details.

``--replication_method <replication method>``
    Optional replication method. Defaults to ``replication method``, which is
    the only replication method currently implemented in pyDCOP.

``--ktarget <resiliency_level>``
    Optional replication level (aka number of replicas for each computation).
    Defaults to 3

``--scenario <scenario_files>``
  Path to the files containing the scenario.
  :ref:`yaml definition<usage_file_formats_scenario>` for the format.

``<dcop_files>``
  One or several paths to the files containing the dcop. If several paths are
  given, their content is concatenated as used a the
  :ref:`yaml definition<usage_file_formats_dcop>` for the
  DCOP.



Examples
--------

Runnig the DCOP from the file ``dcop.yaml``, using the initial ditribution from
``dist.yaml`` ::

    pydcop -v 2 run --algo dsa  \\
                    --distribution dist.yaml \\
                    --scenario scenario.yaml \\
                    --collect_on period \\
                    --period 1 \\
                    --run_metrics run_dcop.csv \\
                    dcop.yaml

"""
import json
import logging
import multiprocessing
import threading
import traceback
from functools import partial
from queue import Queue, Empty
from threading import Thread
import numpy as np

import sys

from pydcop.algorithms import list_available_algorithms
from pydcop.commands._utils import (
    _error,
    prepare_metrics_files,
    _load_modules,
    build_algo_def,
    collect_tread,
    add_csvline,
)
from pydcop.dcop.dcop import filter_dcop
from pydcop.dcop.yamldcop import load_dcop_from_file, load_scenario_from_file
from pydcop.distribution.yamlformat import load_dist_from_file
from pydcop.infrastructure.run import run_local_thread_dcop, run_local_process_dcop
from pydcop.replication.yamlformat import load_replica_dist, load_replica_dist_from_file

logger = logging.getLogger("pydcop.cli.run")


def set_parser(subparsers):

    algorithms = list_available_algorithms()
    logger.debug("Available DCOP algorithms %s", algorithms)
    parser = subparsers.add_parser("run", help="run a dcop")

    parser.set_defaults(func=run_cmd)
    parser.set_defaults(on_timeout=on_timeout)
    parser.set_defaults(on_force_exit=on_force_exit)

    parser.add_argument("dcop_files", type=str, nargs="+", help="dcop file")

    parser.add_argument(
        "-a",
        "--algo",
        required=True,
        choices=algorithms,
        help="algorithm for solving the dcop",
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
        required=True,
        help="distribution of the computations on agents, " "as a yaml file ",
    )

    # FIXME: allow loading replica dist from file and pass it to the
    # orchestrator
    # parser.add_argument('-r', '--replica_dist',
    #                    help='distribution of the computations replicas on '
    #                         'agents, as a yaml file ')

    parser.add_argument(
        "-r",
        "--replication_method",
        default="dist_ucs_hostingcosts",
        help="replication method",
    )
    parser.add_argument(
        "-k", "--ktarget", default=3, type=int, help="Requested resiliency level"
    )

    parser.add_argument("-s", "--scenario", required=True, help="scenario file")

    parser.add_argument(
        "-m",
        "--mode",
        default="thread",
        choices=["thread", "process"],
        help="run agents as threads or processes",
    )

    # Statistics collection arguments:
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

    # TODO : remove, this should no be at this level
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


dcop = None
orchestrator = None
INFINITY = None

collect_on = None
run_metrics = None
end_metrics = None

timeout_stopped = False
output_file = None

DISTRIBUTION_METHODS = ["oneagent", "adhoc", "ilp_fgdp", "heur_comhost", "oilp_secp_fgdp", "gh_secp_fgdp", "gh_secp_cgdp", "oilp_cgdp", "gh_cgdp"]

def run_cmd(args, timer=None, timeout=None):
    logger.debug('dcop command "run" with arguments {}'.format(args))

    global INFINITY, collect_on, output_file
    INFINITY = args.infinity
    collect_on = args.collect_on
    output_file = args.output

    period = None
    if args.collect_on == "period":
        period = 1 if args.period is None else args.period
    else:
        if args.period is not None:
            _error('Cannot use "period" argument when collect_on is not ' '"period"')

    csv_cb = prepare_metrics_files(args.run_metrics, args.end_metrics, collect_on)

    _, algo_module, graph_module = _load_modules(None, args.algo)

    global dcop
    logger.info("loading dcop from {}".format(args.dcop_files))
    dcop = load_dcop_from_file(args.dcop_files)

    dcop = filter_dcop(dcop)

    if args.distribution in DISTRIBUTION_METHODS:
        dist_module, algo_module, graph_module = _load_modules(
            args.distribution, args.algo
        )
    else:
        dist_module, algo_module, graph_module = _load_modules(None, args.algo)

    logger.info("loading scenario from {}".format(args.scenario))
    scenario = load_scenario_from_file(args.scenario)

    logger.info("Building computation graph ")
    cg = graph_module.build_computation_graph(dcop)

    logger.info("Distributing computation graph ")
    if dist_module is not None:
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
            replication=args.replication_method,
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
        )

    orchestrator.set_error_handler(_orchestrator_error)

    try:
        orchestrator.deploy_computations()
        orchestrator.start_replication(args.ktarget)
        if orchestrator.wait_ready():
            orchestrator.run(scenario, timeout=timeout)
            if timer:
                timer.cancel()
            if not timeout_stopped:
                if orchestrator.status == "TIMEOUT":
                    _results("TIMEOUT")
                    sys.exit(0)
                elif orchestrator.status != "STOPPED":
                    _results("FINISHED")
                    sys.exit(0)

    except Exception as e:
        logger.error(e, exc_info=1)
        print(e)
        for th in threading.enumerate():
            print(th)
            traceback.print_stack(sys._current_frames()[th.ident])
            print()
        orchestrator.stop_agents(5)
        orchestrator.stop()
        _results("ERROR")


def _orchestrator_error(e):
    print("Error in orchestrator: \n ", e)
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
    else:
        print(json.dumps(metrics, sort_keys=True, indent="  ", cls=NumpyEncoder))


def on_timeout():
    if orchestrator is None:
        return
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
    orchestrator.stop_agents(20)
    orchestrator.stop()
    _results("TIMEOUT")
    sys.exit(0)


def on_force_exit(sig, frame):
    if orchestrator is None:
        return
    orchestrator.status = "STOPPED"
    orchestrator.stop_agents(5)
    orchestrator.stop()
    _results("STOPPED")
    # for th in threading.enumerate():
    #     print(th)
    #     traceback.print_stack(sys._current_frames()[th.ident])
    #     print()
