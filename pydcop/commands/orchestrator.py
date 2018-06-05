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

:ref:`pydcop_commands_agent`


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

``--address <ip_address>``
  Optional IP address the orchestrator will listen on.
  If not given we try to use the primary IP address.

``--port <port>``
  Optional port the orchestrator will listen on.
  If not given we try to use port 9000.

``<dcop_files>``
  One or several paths to the files containing the dcop. If several paths are
  given, their content is concatenated as used a the yaml definition for the
  DCOP.

Examples
--------

Running an orchestrator for 5 seconds (on default IP and port),
to solve a graph coloring DCOP with ``maxsum``.
Computations are distributed
using the  ``adhoc`` algorithm::

  pydcop --timeout 5 orchestrator -a maxsum -d adhoc graph_coloring.yaml


"""

import json
import logging
import sys

import multiprocessing
from importlib import import_module
from time import time

from pydcop.algorithms import list_available_algorithms
from pydcop.commands._utils import build_algo_def
from pydcop.dcop.yamldcop import load_dcop_from_file
from pydcop.distribution.yamlformat import load_dist_from_file
from pydcop.infrastructure.communication import HttpCommunicationLayer
from pydcop.infrastructure.orchestrator import Orchestrator

logger = logging.getLogger('pydcop.cli.orchestrator')


def set_parser(subparsers):
    algorithms = list_available_algorithms()
    logger.debug('Available DCOP algorithms %s', algorithms)

    parser = subparsers.add_parser('orchestrator',
                                   help='run a standalone orchestrator')
    parser.set_defaults(func=run_cmd)
    parser.set_defaults(on_timeout=on_timeout)
    parser.set_defaults(on_force_exit=on_force_exit)

    parser.add_argument('dcop_files', type=str, nargs='+',
                        help="dcop file(s)")
    parser.add_argument('-a', '--algo',
                        choices=algorithms,
                        help='algorithm for solving the dcop')

    parser.add_argument('-p', '--algo_params',
                        type=str,  action='append',
                        help='Optional parameters for the algorithm, given as '
                             'name:value. Use this option several times '
                             'to set several parameters.')

    parser.add_argument('--address', type=str, default=None,
                        help="IP address the orchestrator will listen on. If "
                             "not given we try to use the primary IP address.")

    parser.add_argument('--port', type=int, default=None,
                        help="Port the orchestrator will listen on. If "
                             "not given we try to use port 9000.")

    parser.add_argument('-d', '--distribution',
                        default='oneagent',
                        choices=['oneagent', 'adhoc', 'ilp_fgdp'],
                        help='A yaml file with the distribution or algorithm '
                             'for distributing the computation graph, if not '
                             'given the `oneagent` will be used (one '
                             'computation for each agent)')


orchestrator = None
start_time = 0


def run_cmd(args):
    logger.debug('dcop command "solve" with arguments {} '.format(args))

    dcop_yaml_files = args.dcop_files

    if args.distribution in ['oneagent', 'adhoc', 'ilp_fgdp']:
        dist_module, algo_module, graph_module = _load_modules(
            args.distribution, args.algo)
    else:
        dist_module, algo_module, graph_module = _load_modules(None,
                                                               args.algo)

    logger.info('loading dcop from {}'.format(dcop_yaml_files))
    dcop = load_dcop_from_file(dcop_yaml_files)

    # Build factor-graph computation graph
    logger.info('Building computation graph for dcop {}'
                .format(dcop_yaml_files))
    cg = graph_module.build_computation_graph(dcop)

    logger.info('Distributing computation graph ')
    if dist_module is not None:
        distribution = dist_module.distribute(
            cg, dcop.agents.values(),
            hints=dcop.dist_hints,
            computation_memory=algo_module.computation_memory,
            communication_load=algo_module.communication_load)
    else:
        distribution = load_dist_from_file(args.distribution)
        logger.debug('Distribution Computation graph: %s ', distribution)

    logger.info('Dcop distribution : {}'.format(distribution))

    algo = build_algo_def(algo_module, args.algo, dcop.objective,
                            args.algo_params)

    # When using the (default) 'fork' start method, http servers on agent's
    # processes do not work (why ?)
    multiprocessing.set_start_method('spawn')

    # FIXME
    infinity = 10000

    global orchestrator, start_time
    port = args.port if args.port else 9000
    addr = args.address if args.address else None
    comm = HttpCommunicationLayer((addr, port))
    orchestrator = Orchestrator(algo, cg, distribution, comm, dcop,
                                infinity)

    start_time = time()
    orchestrator.start()
    orchestrator.deploy_computations()
    orchestrator.run()
    # orchestrator.join()


def on_force_exit(sig, frame):
    print('FORCE EXIT')
    orchestrator.stop_agents(10)
    orchestrator.stop()
    assignment, cost, duration = _results()

    output = {
        'status': 'STOPPED',
        'assignment': assignment,
        'costs':  cost,
        'duration': duration
    }
    print(json.dumps(output, sort_keys=True, indent='  '))


def _results():
    sol = orchestrator.current_solution()[0]
    assignment = {k: sol[k][0] for k in sol if sol[k]}
    cost = orchestrator.current_global_cost()[0]
    duration = time() - start_time
    return assignment, cost, duration


def on_timeout():
    orchestrator.stop_agents()
    orchestrator.stop()
    assignment, cost, duration = _results()
    output = {
        'status': 'TIMEOUT',
        'assignment': assignment,
        'costs':  cost,
        'duration': duration
    }
    print(json.dumps(output, sort_keys=True, indent='  '))


def _load_modules(dist, algo):
    dist_module, algo_module, graph_module = None, None, None
    try:
        dist_module = import_module('pydcop.distribution.{}'.format(dist))
        # TODO check the imported module has the right methods ?
    except ImportError:
        _error('Could not find distribution method {}'.format(dist))

    try:
        algo_module = import_module('pydcop.algorithms.{}'.format(algo))
        # TODO check the imported module has the right methods ?

        graph_module = import_module('pydcop.computations_graph.{}'.
                                     format(algo_module.GRAPH_TYPE))
    except ImportError:
        _error('Could not find computation graph type: {}'.format(
            algo_module.GRAPH_TYPE))

    return dist_module, algo_module, graph_module


def _error(msg):
    print('Error: {}'.format(msg))
    sys.exit(2)
