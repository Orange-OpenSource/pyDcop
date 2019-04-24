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

    def test_oneagent_pseudotree(self):
        result = run_distribute('graph_coloring1.yaml', 'oneagent',
                                'pseudotree')
        dist = result['distribution']

        # With oneagent distribution, each agent host at most one computation
        for a in dist:
            self.assertLessEqual(len(dist[a]), 1)

        self.assertTrue(is_hosted(dist, 'v1'))
        self.assertTrue(is_hosted(dist, 'v2'))
        self.assertTrue(is_hosted(dist, 'v3'))

    def test_oneagent_factorgraph(self):
        result = run_distribute('graph_coloring1.yaml', 'oneagent',
                                'factor_graph')
        dist = result['distribution']

        # With oneagent distribution, each agent host at most one computation
        for a in dist:
            self.assertLessEqual(len(dist[a]), 1)

        self.assertTrue(is_hosted(dist, 'v1'))
        self.assertTrue(is_hosted(dist, 'v2'))
        self.assertTrue(is_hosted(dist, 'v3'))

    def test_oneagent_constraints_hypergraph(self):
        result = run_distribute('graph_coloring1.yaml', 'oneagent',
                                'constraints_hypergraph')
        dist = result['distribution']

        # With oneagent distribution, each agent host at most one computation
        for a in dist:
            self.assertLessEqual(len(dist[a]), 1)

        self.assertTrue(is_hosted(dist, 'v1'))
        self.assertTrue(is_hosted(dist, 'v2'))
        self.assertTrue(is_hosted(dist, 'v3'))

    @unittest.skip('dpop does not define computation size')
    def test_adhoc_pseudotree(self):
        result = run_distribute('graph_coloring1.yaml', 'adhoc',
                                'pseudotree', algo='dpop')
        dist = result['distribution']

        self.assertTrue(is_hosted(dist, 'v1'))
        self.assertTrue(is_hosted(dist, 'v2'))
        self.assertTrue(is_hosted(dist, 'v3'))

    def test_adhoc_factorgraph(self):
        result = run_distribute('graph_coloring1.yaml', 'adhoc',
                                'factor_graph', algo='maxsum')
        dist = result['distribution']

        self.assertTrue(is_hosted(dist, 'v1'))
        self.assertTrue(is_hosted(dist, 'v2'))
        self.assertTrue(is_hosted(dist, 'v3'))

    def test_adhoc_constraints_hypergraph(self):
        result = run_distribute('graph_coloring1.yaml', 'adhoc',
                                'constraints_hypergraph', algo='dsa')
        dist = result['distribution']

        self.assertTrue(is_hosted(dist, 'v1'))
        self.assertTrue(is_hosted(dist, 'v2'))
        self.assertTrue(is_hosted(dist, 'v3'))

    def test_ilp_fgdp_factorgraph(self):
        # When using ilp-fgdp, we must also provide the algorithm
        result = run_distribute('graph_coloring1.yaml', 'ilp_fgdp',
                                'factor_graph', algo='maxsum')
        dist = result['distribution']

        # ILP-FGDP requires that each agent hosts at least one computation
        # and we have eaxctly 5 computations and 5 agents here:
        for a in dist:
            self.assertEqual(len(dist[a]), 1)

        self.assertTrue(is_hosted(dist, 'v1'))
        self.assertTrue(is_hosted(dist, 'v2'))
        self.assertTrue(is_hosted(dist, 'v3'))

    def test_ilp_fgdp_pseudotree(self):
        # ilp-fgdp does not work with pseudo-tree graph model
        self.assertRaises(CalledProcessError, run_distribute,
                          'graph_coloring1.yaml', 'ilp_fgdp',
                          'pseudotree', algo='maxsum')

    def test_ilp_fgdp_constraints_hypergraph(self):
        # ilp-fgdp does not work with pseudo-tree graph model
        self.assertRaises(CalledProcessError, run_distribute,
                          'graph_coloring1.yaml', 'ilp_fgdp',
                          'constraints_hypergraph', algo='maxsum')

    def test_ilp_compref_factorgraph(self):
        # When using ilp-compref, we must also provide the algorithm
        result = run_distribute('graph_coloring1.yaml', 'ilp_compref',
                                'factor_graph', algo='maxsum')
        # lame: we do not check the result, we just ensure we do not crash

    def test_ilp_compref_constraints_hypergraph(self):
        result = run_distribute('graph_coloring1.yaml', 'ilp_compref',
                                'constraints_hypergraph', algo='dsa')
        # lame: we do not check the result, we just ensure we do not crash


class DistAlgoOpionCompatibility(unittest.TestCase):

    def test_dist_with_only_algo_only(self):
        run_distribute('graph_coloring1.yaml', 'oneagent',
                                algo='maxsum')

    def test_dist_with_graph_only(self):
        run_distribute('graph_coloring1.yaml', 'oneagent',
                                graph='factor_graph')

    def test_incompatible_graph_algo_must_fail(self):
        self.assertRaises(CalledProcessError, run_distribute,
                          'graph_coloring1.yaml', 'oneagent',
                          algo='dsa', graph='factor_graph')

def is_hosted(mapping, computation: str):
    for a in mapping:
        if computation in mapping[a]:
            return True
    return False


def run_distribute(filename, distribution, graph=None, algo=None):
    """
    Run the distribute cli command with the given parameters
    """
    filename = instance_path(filename)
    algo_opt = '' if algo is None else '-a ' + algo
    graph_opt = '' if graph is None else '-g ' + graph
    cmd = 'pydcop distribute -d {distribution} {graph_opt} ' \
          '{algo_opt} {file}'.format(distribution=distribution,
                                     graph_opt=graph_opt,
                                     algo_opt=algo_opt,
                                     file=filename)
    output = check_output(cmd, stderr=STDOUT, timeout=10, shell=True)
    return yaml.load(output.decode(encoding='utf-8'), Loader=yaml.FullLoader)
