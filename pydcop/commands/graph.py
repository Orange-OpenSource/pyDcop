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


import logging
from importlib import import_module
import sys
import yaml

from pydcop.dcop.yamldcop import load_dcop_from_file

logger = logging.getLogger('pydcop.cli.graph')


# TODO : ass more graph metrics:
# * number of cycles
# * is connected ?
# * number of sub-graph (if not connected)

def set_parser(subparsers):

    parser = subparsers.add_parser('graph',
                                   help='Graph metrics for dcop graphs')
    parser.set_defaults(func=run_cmd)

    parser.add_argument('dcop_file', type=str, nargs='+', help="dcop file")

    parser.add_argument('-g', '--graph',
                        choices=['factor_graph', 'pseudotree',
                                 'constraints_hypergraph'],
                        help='graphical model for dcop computations')


def run_cmd(args):
    logger.debug('dcop command "graph" with arguments {} '.format(args))

    dcop_yaml_file = args.dcop_file
    logger.info('loading dcop from {}'.format(dcop_yaml_file))
    dcop = load_dcop_from_file(dcop_yaml_file)

    try:
        graph_module = import_module('pydcop.computations_graph.{}'.
                                     format(args.graph))
        graph_stats(dcop, graph_module)
    except ImportError:
        _error('Could not find computation graph type: {}'.format(
            args.graph))


def graph_stats(dcop, graph_module):

    # Build factor-graph computation graph
    logger.info('Building computation graph for dcop {}'
                .format(dcop.name))
    cg = graph_module.build_computation_graph(dcop)

    edges_count = len(list(cg.links))
    nodes_count = len(list(cg.nodes))
    density = cg.density()

    # TODO: add other graph metrics :
    # branching factor
    # diameter
    # number or loops
    # root (when it's a tree)
    # # variables and # factors, when it's a factor graph

    # Note : when using variables with integrated costs, the costs factors
    # are not accounted for in the metrics.

    result = {
        'status': 'OK',
        'variables_count': len(dcop.variables),
        'constraints_count': len(dcop.constraints),
        'nodes_count': nodes_count,
        'edges_count':  edges_count,
        'density': density
    }
    print(yaml.dump(result, default_flow_style=False))


def _error(msg):
    print('Error: {}'.format(msg))
    sys.exit(2)
