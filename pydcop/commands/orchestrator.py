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

"""
The 'orchestrator' dcop cli command runs an orchestrator, which waits for 
agents, deploy 
computations and receive selected values from agents. 
Agents must be run separately using the 'agent command'.

The 'orchestrator' command support:
 * the global timeout argument : stop all registered agent and display the 
   current result
 * the forced interruption (ctrl-c) : same as timeout

"""


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
                        type=str,
                        nargs='*',
                        help='parameters for the algorithm , given as '
                             'name:value. Several parameter can be given.')

    parser.add_argument('-d', '--distribution',
                        choices=['oneagent', 'adhoc', 'ilp_fgdp'],
                        help='algorithm for distributing the computation '
                             'graph')


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
    port = 9000
    comm = HttpCommunicationLayer(('127.0.0.1', port))
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
