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


import unittest
from subprocess import check_output, STDOUT, CalledProcessError

import yaml

from tests.dcop_cli.utils import instance_path


class GraphColoring1(unittest.TestCase):

    def test_pseudotree(self):
        result = run_graph('graph_coloring1.yaml', 'pseudotree')

        self.assertEqual(result['nodes_count'], 3)
        self.assertEqual(result['edges_count'], 4)
        self.assertEqual(result['density'], 4/(3*2))

    def test_factor_graph(self):
        result = run_graph('graph_coloring1.yaml', 'factor_graph')

        self.assertEqual(result['nodes_count'], 5)
        self.assertEqual(result['edges_count'], 4)
        self.assertEqual(result['density'], 0.4)

    def test_constraints_hypergraph(self):
        result = run_graph('graph_coloring1.yaml', 'constraints_hypergraph')

        self.assertEqual(result['nodes_count'], 3)
        self.assertEqual(result['edges_count'], 2)
        self.assertEqual(result['density'], 2/3)


class SecpSimple1(unittest.TestCase):

    def test_pseudotree(self):
        result = run_graph('secp_simple1.yaml', 'pseudotree')

        self.assertEqual(result['nodes_count'], 4)
        # In secp _simple1, the model depends on all llights, meaning there
        # is a clique and the grpah is fully connected.
        self.assertEqual(result['edges_count'], 12)
        self.assertEqual(result['density'], 1)

    def test_factor_graph(self):
        result = run_graph('secp_simple1.yaml', 'factor_graph')

        self.assertEqual(result['nodes_count'], 6)
        self.assertEqual(result['edges_count'], 6)
        self.assertEqual(result['density'], 2* 6 / (6 * 5 ))

    def test_constraints_hypergraph(self):
        result = run_graph('secp_simple1.yaml', 'constraints_hypergraph')

        self.assertEqual(result['nodes_count'], 4)
        self.assertEqual(result['edges_count'], 2)
        self.assertEqual(result['density'], 1/3)


def run_graph(filename, graph):
    filename = instance_path(filename)
    cmd = 'pydcop graph -g {graph} {file}'.format(graph=graph,
                                                   file=filename)
    output = check_output(cmd, stderr=STDOUT, timeout=10, shell=True)
    return yaml.load(output.decode(encoding='utf-8'), Loader=yaml.FullLoader)
