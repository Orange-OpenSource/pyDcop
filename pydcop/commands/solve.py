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
               <dcop_files>


Description
-----------

The ``solve`` command is a shorthand for the ``agent`` and ``orchestrator``
commands,
it solves a DCOP on the local machine, automatically creating the
orchestrator and the agents as specified in the dcop definitions.

Using ``solve`` all agents run in the same machine and using ``--mode thread``
communicate in memory (without network), which scales easily to more than
100 agents.

Depending on the ``--mode`` parameter, agents will be
created as threads (lightweight) or as process (heavier, but better
parallelism on a multi-core cpu).

Notes
-----

Depending on the DCOP algorithm, the solve process may or may not automatically.
For example, :ref:`DPOP<implementation_reference_algorithms_dpop>`
has a clear termination condition and the command will return once this
condition is reached.
On the other hand, some other algorithm like
:ref:`MaxSum<implementation_reference_algorithms_maxsum>` have
no clear termination condition
(several options are available,
which could be passed as argument to the algorithm).

For these algorithm you need to use the global
``--timeout`` option.
You can also stop the process manually with ``CTRL+C``.

Agents are only stopped once they have handled all messages already
present in their message queue : this means that even a force stop can take
some time.

Options
-------

``--algo <dcop_algorithm>`` / ``-a <dcop_algorithm>``
  Name of the dcop algorithm, e.g. 'maxsum', 'dpop', 'dsa', etc.

``--algo_params <params>`` / ``-p <params>``
  Parameters (optional) for the DCOP algorithm, given as string "name:value".
  May be used multiple times to set several parameters. Available parameters
  depend on the algorithm, check algorithms documentation.

``--distribution <distribution>`` / ``-d <distribution>``
  Either a distribution algorithm ('oneagent', 'adhoc', 'ilp_fgdp', etc.) or
  the path to a yaml file containing the distribution

``--mode <mode>`` / ``-m``
    Indicated if agents must be run as threads (default) or processes.
    either ``'thread'`` or ``'process'``

``--collect_on <collect_mode>`` / ``-c``
    Metric collection mode, one of ``'value_change'``, ``'cycle_change'``,
    ``'period'``.

``--period <p>``
    When using ``--collect_on period``, the period in second for metrics
    collection.

``--run_metrics <file>``
    File to store store metrics.

``--end_metrics <file>``
    End metrics (i.e. when the solve process stops) will be appended to this
    file.

``<dcop_files>``
  One or several paths to the files containing the dcop. If several paths are
  given, their content is concatenated as used a the yaml definition for the
  DCOP.


Examples
--------

The simplest form is to simply specify an algorithm and a dcop yaml file.
Beware that, depending on the algorithm, this command may never return and
need to be stopped with CTRL+C::

    dcop.py solve --algo maxsum  graph_coloring1.yaml
    dcop.py -t 5 solve --algo maxsum  graph_coloring1.yaml


"""

import json
import logging
import os
import multiprocessing
import sys
from functools import partial
from queue import Queue, Empty
from threading import Thread

from pydcop.algorithms import list_available_algorithms
from pydcop.commands._utils import build_algo_def, _error, _load_modules
from pydcop.dcop.yamldcop import load_dcop_from_file
from pydcop.distribution.yamlformat import load_dist_from_file
from pydcop.infrastructure.run import run_local_thread_dcop, \
    run_local_process_dcop


logger = logging.getLogger('pydcop.cli.solve')


def set_parser(subparsers):

    algorithms = list_available_algorithms()
    logger.debug('Available DCOP algorithms %s', algorithms)

    parser = subparsers.add_parser('solve',
                                   help='solve static dcop')
    parser.set_defaults(func=run_cmd)
    parser.set_defaults(on_timeout=on_timeout)
    parser.set_defaults(on_force_exit=on_force_exit)

    parser.add_argument('dcop_files', type=str,  nargs='+',
                        help="The DCOP, in one or several yaml file(s)")

    parser.add_argument('-a', '--algo',
                        choices=algorithms, required=True,
                        help='The algorithm for solving the dcop')
    parser.add_argument('-p', '--algo_params',
                        type=str,  action='append',
                        help='Optional parameters for the algorithm, given as '
                             'name:value. Use this option several times '
                             'to set several parameters.')

    parser.add_argument('-d', '--distribution', type=str,
                        default='oneagent',
                        help='A yaml file with the distribution or algorithm '
                             'for distributing the computation graph, if not '
                             'given the `oneagent` will be used (one '
                             'computation for each agent)')
    parser.add_argument('-m', '--mode',
                        default='thread',
                        choices=['thread', 'process'],
                        help='run agents as threads or processes')

    parser.add_argument('-c', '--collect_on',
                        choices=['value_change', 'cycle_change', 'period'],
                        default='value_change',
                        help='When should a "new" assignment be observed')

    parser.add_argument('--period', type=float,
                        default=None,
                        help='Period for collecting metrics. only available '
                             'when using --collect_on period. Defaults to 1 '
                             'second if not specified')

    parser.add_argument('--run_metrics', type=str,
                        default=None,
                        help="Use this option to regularly store the data "
                             "in a csv file.")

    parser.add_argument('--end_metrics', type=str,
                        default=None,
                        help="Use this option to append the metrics of the "
                             "end of the run to a csv file.")

    parser.add_argument('--infinity', '-i', default=float('inf'),
                        type=float,
                        help='Argument to determine the value used for '
                             'infinity in case of hard constraints, '
                             'for algorithms that do not use symbolic '
                             'infinity. Defaults to 10 000')


dcop = None
orchestrator = None
INFINITY = None

# Files for logging metrics
columns = {
    'cycle_change': ['cycle', 'time', 'cost', 'violation', 'msg_count',
                     'msg_size',
                     'active_ratio', 'status'],
    'value_change': ['time', 'cycle', 'cost', 'violation', 'msg_count',
                     'msg_size', 'active_ratio', 'status'],
    'period': ['time', 'cycle', 'cost', 'violation', 'msg_count', 'msg_size',
               'active_ratio', 'status']
}

collect_on = None
run_metrics = None
end_metrics = None

timeout_stopped = False
output_file = None


def add_csvline(file, mode, metrics):
    data = [metrics[c] for c in columns[mode]]
    line = ','.join([str(d) for d in data])

    with open(file, mode='at', encoding='utf-8') as f:
        f.write(line)
        f.write('\n')


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
        elif not os.path.exists(os.path.dirname(run_metrics)):
            os.makedirs(os.path.dirname(run_metrics))
        # Add column labels in file:
        headers = ','.join(columns[mode])
        with open(run_metrics, 'w', encoding='utf-8') as f:
            f.write(headers)
            f.write('\n')
        csv_cb = partial(add_csvline, run_metrics, mode)
    else:
        csv_cb = None

    if end is not None:
        end_metrics = end
        if not os.path.exists(os.path.dirname(end_metrics)):
            os.makedirs(os.path.dirname(end_metrics))
        # Add column labels in file:
        if not os.path.exists(end_metrics):
            headers = ','.join(columns[mode])
            with open(end_metrics, 'w', encoding='utf-8') as f:
                f.write(headers)
                f.write('\n')

    return csv_cb


def run_cmd(args, timer=None):
    logger.debug('dcop command "solve" with arguments {}'.format(args))

    global INFINITY, collect_on, output_file
    INFINITY = args.infinity
    output_file = args.output
    collect_on = args.collect_on

    period = None
    if args.collect_on == 'period':
        period = 1 if args.period is None else args.period
    else:
        if args.period is not None:
            _error('Cannot use "period" argument when collect_on is not '
                   '"period"')

    csv_cb = prepare_metrics_files(args.run_metrics, args.end_metrics,
                                   collect_on)

    if args.distribution in ['oneagent', 'adhoc', 'ilp_fgdp']:
        dist_module, algo_module, graph_module = _load_modules(args.distribution,
                                                               args.algo)
    else:
        dist_module, algo_module, graph_module = _load_modules(None,
                                                               args.algo)

    global dcop
    logger.info('loading dcop from {}'.format(args.dcop_files))
    dcop = load_dcop_from_file(args.dcop_files)

    # Build factor-graph computation graph
    logger.info('Building computation graph ')
    cg = graph_module.build_computation_graph(dcop)
    logger.debug('Computation graph: %s ', cg)

    logger.info('Distributing computation graph ')
    if dist_module is not None:
        distribution = dist_module.\
            distribute(cg, dcop.agents.values(),
                       hints=dcop.dist_hints,
                       computation_memory=algo_module.computation_memory,
                       communication_load=algo_module.communication_load)
    else:
        distribution = load_dist_from_file(args.distribution)
    logger.debug('Distribution Computation graph: %s ', distribution)

    logger.info('Dcop distribution : {}'.format(distribution))

    algo = build_algo_def(algo_module, args.algo, dcop.objective,
                            args.algo_params)

    # Setup metrics collection
    collector_queue = Queue()
    collect_t = Thread(target=collect_tread,
                       args=[collector_queue, csv_cb],
                       daemon=True)
    collect_t.start()

    global orchestrator
    if args.mode == 'thread':
        orchestrator = run_local_thread_dcop(algo, cg, distribution, dcop,
                                             INFINITY,
                                             collector=collector_queue,
                                             collect_moment=args.collect_on,
                                             period=period)
    elif args.mode == 'process':

        # Disable logs from agents, they are in other processes anyway
        agt_logs = logging.getLogger('pydcop.agent')
        agt_logs.disabled = True

        # When using the (default) 'fork' start method, http servers on agent's
        # processes do not work (why ?)
        multiprocessing.set_start_method('spawn')
        orchestrator = run_local_process_dcop(algo, cg, distribution, dcop,
                                              INFINITY,
                                              collector=collector_queue,
                                              collect_moment=args.collect_on,
                                              period=period)

    try:
        orchestrator.deploy_computations()
        orchestrator.run()
        if timer:
            timer.cancel()
        if not timeout_stopped:
            _results('FINISHED')
            sys.exit(0)

        # in case it did not stop, dump remaining threads

    except Exception as e:
        logger.error(e, exc_info=1)
        orchestrator.stop_agents(5)
        orchestrator.stop()
        _results('ERROR')


def on_timeout():
    if orchestrator is None:
        return
    global timeout_stopped
    timeout_stopped = True
    # Stopping agents can be rather long, we need a big timeout !
    orchestrator.stop_agents(20)
    orchestrator.stop()
    _results('TIMEOUT')
    sys.exit(0)


def on_force_exit(sig, frame):
    if orchestrator is None:
        return
    orchestrator.stop_agents(5)
    orchestrator.stop()
    _results('STOPPED')


def _results(status):
    """
    Outputs results and metrics on stdout and trace last metrics in csv
    files if requested.

    :param status:
    :return:
    """

    metrics = orchestrator.end_metrics()
    metrics['status'] = status
    global end_metrics, run_metrics
    if end_metrics is not None:
        add_csvline(end_metrics, collect_on, metrics)
    if run_metrics is not None:
        add_csvline(run_metrics, collect_on, metrics)

    if output_file:
        with open(output_file, encoding='utf-8', mode='w') as fo:
                fo.write(json.dumps(metrics, sort_keys=True, indent='  '))

    print(json.dumps(metrics, sort_keys=True, indent='  '))




