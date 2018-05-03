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

Distribute a dcop over a set of agents.

Print the distribution obtained.

Synopsis
--------
::

    pydcop distribute --graph <graph-model> --dist <distribution method>
        --algo <dcop-algorithm> <dcop-files>


Options
-------

* ``--graph / -g <graph-model>`` : computation graph model

* ``--dist / -d <distribution method>`` : distribution method

* ``--algo / -a <dcop-algorithm>`` : the algorithm whose computation must be
  distributed

* ``<dcop-files>`` : one or several files containing the dcop


Usage Examples
--------------

* distribute a dcop solved with dsa, which runs on a constraint hyper-graph,
  using the ``ilp_compref`` distribution method::

    pydcop distribute -g constraints_hypergraph -d ilp_compref -a dsa \\
      tests/instances/graph_coloring_10_4_15_0.1_capa_costs.yml


"""

import logging
from importlib import import_module
import sys
import yaml

from pydcop.algorithms import list_available_algorithms
from pydcop.commands._utils import _error
from pydcop.dcop.yamldcop import load_dcop_from_file
from pydcop.distribution.objects import ImpossibleDistributionException

logger = logging.getLogger('pydcop.cli.distribute')


def set_parser(subparsers):

    algorithms = list_available_algorithms()

    parser = subparsers.add_parser('distribute',
                                   help='distribute a static dcop')
    parser.set_defaults(func=run_cmd)

    parser.add_argument('dcop_files', type=str, nargs='+', metavar='FILE',
                        help="dcop file")

    parser.add_argument('-g', '--graph',
                        required=True,
                        choices=['factor_graph', 'pseudotree',
                                 'constraints_hypergraph'],
                        help='graphical model for dcop computations')

    parser.add_argument('-d', '--dist',
                        choices=['oneagent', 'adhoc', 'ilp_fgdp',
                                 'ilp_compref', 'heur_comhost'],
                        required=True,
                        help='algorithm for distributing the computation '
                             'graph')

    parser.add_argument('-a', '--algo',
                        choices=algorithms,
                        required=False,
                        help='Optional, only needed for '
                              'distribution methods that require '
                              'the memory footprint and '
                              'communication load for computations')


def run_cmd(args):
    logger.debug('dcop command "distribute" with arguments {} '.format(args))

    dcop_yaml_files = args.dcop_files
    dist_module, algo_module, graph_module = _load_modules(args.dist,
                                                           args.algo,
                                                           args.graph)

    logger.info('loading dcop from {}'.format(dcop_yaml_files))
    dcop = load_dcop_from_file(dcop_yaml_files)

    # Build factor-graph computation graph
    logger.info('Building computation graph for dcop {}'
                .format(dcop_yaml_files))
    cg = graph_module.build_computation_graph(dcop)

    logger.info('Distributing computation graph for dcop {}'
                .format(dcop_yaml_files))

    if algo_module is None:
        computation_memory = None
        communication_load = None
    else:
        computation_memory = algo_module.computation_memory
        communication_load = algo_module.communication_load

    try:
        distribution = dist_module\
            .distribute(cg, dcop.agents.values(),
                        hints=dcop.dist_hints,
                        computation_memory=computation_memory,
                        communication_load=communication_load)
        dist = distribution.mapping()
        cost = dist_module.distribution_cost(
            distribution, cg, dcop.agents.values(),
            computation_memory=computation_memory,
            communication_load=communication_load)

        result = {
            'inputs': {
                'dist_algo': args.dist,
                'dcop': args.dcop_files,
                'graph': args.graph,
                'algo': args.algo,
            },
            'distribution': dist,
            'cost': cost
        }
        if args.output is not None:
            with open(args.output, encoding='utf-8', mode='w') as fo:
                fo.write(yaml.dump(result))
        print(yaml.dump(result))
        sys.exit(0)

    except ImpossibleDistributionException as e:
        result = {
            'status': 'FAIL',
            'error': str(e)
        }
        print(yaml.dump(result))
        sys.exit(2)


def _load_modules(dist, algo, graph):
    dist_module, algo_module, graph_module = None, None, None
    try:
        dist_module = import_module('pydcop.distribution.{}'.format(dist))
        # TODO check the imported module has the right methods ?
    except ImportError as e:
        _error('Could not find distribution method {}'.format(dist), e)

    try:
        # Algo is optional, do not fail if we cannot find it
        if algo is not None:
            algo_module = import_module('pydcop.algorithms.{}'.format(algo))
        graph_module = import_module('pydcop.computations_graph.{}'.
                                     format(graph))
    except ImportError as e:
        _error('Could not find computation graph type: {}'.format(
            algo_module.GRAPH_TYPE), e)

    return dist_module, algo_module, graph_module
