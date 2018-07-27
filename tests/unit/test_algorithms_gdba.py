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
from unittest.mock import MagicMock

import numpy

from pydcop.algorithms.gdba import GdbaComputation
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import AsNAryFunctionRelation, NAryMatrixRelation, \
    UnaryFunctionRelation, NAryFunctionRelation, generate_assignment_as_dict


class GdbaAlgoTest(unittest.TestCase):
    def test_init_from_constraints_as_functions(self):
        domain = list(range(3))
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            return x1_ + x2_

        g = GdbaComputation(x1, [phi], comp_def=MagicMock())

        m = NAryMatrixRelation.from_func_relation(phi)
        (c_mat, mini, maxi) = g.__constraints__[0]
        self.assertEqual(c_mat, m)
        self.assertEqual(mini, 0)
        self.assertEqual(maxi, 4)

    def test_init_from_constraints_as_matrices(self):
        domain = list(range(2))
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)

        m = numpy.matrix('1 0 ; 0 1')
        mat = NAryMatrixRelation([x1, x2], m)

        g = GdbaComputation(x1, [mat], comp_def=MagicMock())
        c_mat, mini, maxi = g.__constraints__[0]

        self.assertTrue(numpy.array_equal(mat._m, m))
        self.assertEqual(mini, 0)
        self.assertEqual(maxi, 1)


class TestsCostComputation(unittest.TestCase):
    def test_compute_eval_binary(self):
        domain = list(range(2))
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            if x1_ == x2_:
                return 1
            return 0

        g = GdbaComputation(x1, [phi], comp_def=MagicMock())
        g._neighbors_values['x2'] = 0

        eval0, _ = g.compute_eval_value(0)
        eval1, _ = g.compute_eval_value(1)

        self.assertEqual(eval0, 1)
        self.assertEqual(eval1, 0)

    def test_compute_eval_3_ary(self):
        domain = list(range(3))
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            if x1_ == x2_ or x1_ == x3_:
                return 1
            return 0

        g = GdbaComputation(x1, [phi], comp_def=MagicMock())
        g._neighbors_values['x2'] = 0
        g._neighbors_values['x3'] = 1

        eval0, _ = g.compute_eval_value(0)
        eval1, _ = g.compute_eval_value(1)
        eval2, _ = g.compute_eval_value(2)

        self.assertEqual(eval0, 1)
        self.assertEqual(eval1, 1)
        self.assertEqual(eval2, 0)

    def test_min_compute_best_for_binary_constraint(self):
        domain = list(range(2))
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            if x1_ == x2_:
                return 1
            return 0

        g = GdbaComputation(x1, [phi], comp_def=MagicMock())
        g._neighbors_values['x2'] = 0
        bests, best = g._compute_best_improvement()

        self.assertEqual(best, 0)
        self.assertEqual(bests, [1])

    def test_max_compute_best_for_binary_constraint(self):
        domain = list(range(2))
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            if x1_ == x2_:
                return 1
            return 0

        g = GdbaComputation(x1, [phi], mode='max', comp_def=MagicMock())
        g._neighbors_values['x2'] = 1
        bests, best = g._compute_best_improvement()

        self.assertEqual(best, 1)
        self.assertEqual(bests, [1])

    def test_min_compute_best_for_3_ary_constraint(self):
        domain = list(range(3))
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            if x1_ == x2_ or x1_ == x3_:
                return 1
            return 0

        g = GdbaComputation(x1, [phi], comp_def=MagicMock())
        g._neighbors_values['x2'] = 0
        g._neighbors_values['x3'] = 0
        bests, best = g._compute_best_improvement()

        self.assertEqual(best, 0)
        self.assertEqual(bests, [1, 2])

    def test_eff_cost_A_unary(self):
        domain = list(range(3))
        x1 = Variable('x1', domain)

        @AsNAryFunctionRelation(x1)
        def phi(x1_):
            return x1_

        g = GdbaComputation(x1, [phi], comp_def=MagicMock())
        c, _, _ = g.__constraints__[0]
        g.__value__ = 0
        asgt = frozenset({'x1': 0}.items())
        g.__constraints_modifiers__[c][asgt] = 5

        self.assertEqual(g._eff_cost(c, 0), 5)
        self.assertEqual(g._eff_cost(c, 1), 1)
        self.assertEqual(g._eff_cost(c, 2), 2)

    def test_eff_cost_A_n_ary(self):
        domain = list(range(3))
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            if x1_ == x2_:
                return 2
            if x1_ == x3_:
                return 1
            return 0

        g = GdbaComputation(x1, [phi], comp_def=MagicMock())
        g._neighbors_values['x2'] = 1
        g._neighbors_values['x3'] = 2
        c, _, _ = g.__constraints__[0]
        asgt = frozenset({'x1': 0, 'x2': 1, 'x3': 2}.items())
        g.__constraints_modifiers__[c][asgt] = 5

        self.assertEqual(g._eff_cost(c, 0), 5)
        self.assertEqual(g._eff_cost(c, 1), 2)
        self.assertEqual(g._eff_cost(c, 2), 1)

    def test_eff_cost_M_unary(self):
        domain = list(range(3))
        x1 = Variable('x1', domain)

        @AsNAryFunctionRelation(x1)
        def phi(x1_):
            return x1_

        g = GdbaComputation(x1, [phi], modifier='M', comp_def=MagicMock())
        c, _, _ = g.__constraints__[0]
        asgt = frozenset({'x1': 0, }.items())
        g.__constraints_modifiers__[c][asgt] = 5

        self.assertEqual(g._eff_cost(c, 0), 0)
        self.assertEqual(g._eff_cost(c, 1), 1)
        self.assertEqual(g._eff_cost(c, 2), 2)

    def test_eff_cost_M_n_ary(self):
        domain = list(range(3))
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            if x1_ == x2_:
                return 2
            if x1_ == x3_:
                return 1
            return 0

        g = GdbaComputation(x1, [phi], modifier='M', comp_def=MagicMock())
        g._neighbors_values['x2'] = 1
        g._neighbors_values['x3'] = 2
        c, _, _ = g.__constraints__[0]
        asgt = frozenset({'x1': 0, 'x2': 1, 'x3': 2}.items())
        asgt2 = frozenset({'x1': 1, 'x2': 1, 'x3': 2}.items())
        g.__constraints_modifiers__[c][asgt] = 5
        g.__constraints_modifiers__[c][asgt2] = 5

        self.assertEqual(g._eff_cost(c, 0), 0)
        self.assertEqual(g._eff_cost(c, 1), 10)
        self.assertEqual(g._eff_cost(c, 2), 1)


class TestsConstraintViolation(unittest.TestCase):
    domain = list(range(2))
    x1 = Variable('x1', domain)
    x2 = Variable('x2', domain)
    x3 = Variable('x3', domain)

    phi = UnaryFunctionRelation('phi', Variable('x1', domain), lambda x: x)

    phi_n_ary = NAryFunctionRelation(
        lambda x1_, x2_, x3_: 2 if x1_ == x2_ else (1 if x1_ == x3_ else 0),
        [x1, x2, x3])

    def NZ_violation_unary(self):
        g = GdbaComputation(self.x1, [self.phi], comp_def=MagicMock())
        g._neighbors_values['x2'] = 1
        g._neighbors_values['x3'] = 2
        c = g.__constraints__[0]
        self.assertEqual(g._is_violated(c, 0), False)
        self.assertEqual(g._is_violated(c, 1), True)
        self.assertEqual(g._is_violated(c, 2), True)

    def NZ_violation_n_ary(self):
        g = GdbaComputation(self.x1, [self.phi_n_ary], comp_def=MagicMock())
        g._neighbors_values['x2'] = 1
        g._neighbors_values['x3'] = 2
        c = g.__constraints__[0]
        self.assertEqual(g._is_violated(c, 0), False)
        self.assertEqual(g._is_violated(c, 1), True)
        self.assertEqual(g._is_violated(c, 2), True)

    def NM_violation_unary(self):
        g = GdbaComputation(self.x1, [self.phi], comp_def=MagicMock())
        g._neighbors_values['x2'] = 1
        g._neighbors_values['x3'] = 2
        g._violation_mode = 'NM'
        c = g.__constraints__[0]
        self.assertEqual(g._is_violated(c, 0), False)
        self.assertEqual(g._is_violated(c, 1), True)
        self.assertEqual(g._is_violated(c, 2), True)

    def NM_violation_n_ary(self):
        g = GdbaComputation(self.x1, [self.phi_n_ary], comp_def=MagicMock())
        g._neighbors_values['x2'] = 1
        g._neighbors_values['x3'] = 2
        g._violation_mode = 'NM'
        c = g.__constraints__[0]
        self.assertEqual(g._is_violated(c, 0), False)
        self.assertEqual(g._is_violated(c, 1), True)
        self.assertEqual(g._is_violated(c, 2), True)

    def MX_violation_unary(self):
        g = GdbaComputation(self.x1, [self.phi], comp_def=MagicMock())
        g._neighbors_values['x2'] = 1
        g._neighbors_values['x3'] = 2
        g._violation_mode = 'MX'
        c = g.__constraints__[0]
        self.assertEqual(g._is_violated(c, 0), False)
        self.assertEqual(g._is_violated(c, 1), False)
        self.assertEqual(g._is_violated(c, 2), True)

    def MX_violation_n_ary(self):
        g = GdbaComputation(self.x1, [self.phi_n_ary], comp_def=MagicMock())
        g._neighbors_values['x2'] = 1
        g._neighbors_values['x3'] = 2
        g._violation_mode = 'MX'
        c = g.__constraints__[0]
        self.assertEqual(g._is_violated(c, 0), False)
        self.assertEqual(g._is_violated(c, 1), True)
        self.assertEqual(g._is_violated(c, 2), False)


class TestIncreaseCost(unittest.TestCase):
    def test_increase_E(self):
        domain = list(range(2))
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            if x1_ == x2_:
                return 2
            if x1_ == x3_:
                return 1
            return 0

        g = GdbaComputation(x1, [phi], comp_def=MagicMock())
        g.__value__ = 0
        g._neighbors_values['x2'] = 1
        g._neighbors_values['x3'] = 2
        c, _, _ = g.__constraints__[0]
        g._increase_cost(c)
        asgt = {'x1': 0, 'x2': 1, 'x3': 2}
        modifier = frozenset(asgt.items())
        self.assertEqual(g.__constraints_modifiers__[c][modifier], 1)

    def test_increase_R(self):
        domain = list(range(2))
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            if x1_ == x2_:
                return 2
            if x1_ == x3_:
                return 1
            return 0

        g = GdbaComputation(x1, [phi], increase_mode='R', comp_def=MagicMock())
        g._neighbors_values['x2'] = 1
        g._neighbors_values['x3'] = 2
        c, _, _ = g.__constraints__[0]
        g._increase_cost(c)
        asgt = g._neighbors_values.copy()
        for val in x1.domain:
            asgt['x1'] = val
            modifier = frozenset(asgt.items())
            self.assertEqual(g.__constraints_modifiers__[c][modifier], 1)

    def test_increase_C(self):
        domain = list(range(3))
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            if x1_ == x2_:
                return 2
            if x1_ == x3_:
                return 1
            return 0

        g = GdbaComputation(x1, [phi], increase_mode='C', comp_def=MagicMock())
        c, _, _ = g.__constraints__[0]
        g.__value__ = 0
        g._neighbors_values['x2'] = 1
        g._neighbors_values['x3'] = 2
        g._increase_cost(c)
        asgt = {'x1': 0, 'x2': 0, 'x3': 0}
        modifier = frozenset(asgt.items())
        self.assertEqual(g.__constraints_modifiers__[c][modifier], 1)
        asgt = {'x1': 0, 'x2': 0, 'x3': 1}
        modifier = frozenset(asgt.items())
        self.assertEqual(g.__constraints_modifiers__[c][modifier], 1)
        asgt = {'x1': 0, 'x2': 0, 'x3': 2}
        modifier = frozenset(asgt.items())
        self.assertIn(modifier, g.__constraints_modifiers__[c])
        self.assertEqual(g.__constraints_modifiers__[c][modifier], 1)
        asgt = {'x1': 0, 'x2': 1, 'x3': 0}
        modifier = frozenset(asgt.items())
        self.assertEqual(g.__constraints_modifiers__[c][modifier], 1)
        asgt = {'x1': 0, 'x2': 1, 'x3': 1}
        modifier = frozenset(asgt.items())
        self.assertEqual(g.__constraints_modifiers__[c][modifier], 1)
        asgt = {'x1': 0, 'x2': 1, 'x3': 2}
        modifier = frozenset(asgt.items())
        self.assertEqual(g.__constraints_modifiers__[c][modifier], 1)
        asgt = {'x1': 0, 'x2': 2, 'x3': 0}
        modifier = frozenset(asgt.items())
        self.assertEqual(g.__constraints_modifiers__[c][modifier], 1)
        asgt = {'x1': 0, 'x2': 2, 'x3': 1}
        modifier = frozenset(asgt.items())
        self.assertEqual(g.__constraints_modifiers__[c][modifier], 1)
        asgt = {'x1': 0, 'x2': 2, 'x3': 2}
        modifier = frozenset(asgt.items())
        self.assertEqual(g.__constraints_modifiers__[c][modifier], 1)

    def test_increase_T(self):
        domain = list(range(2))
        x1 = Variable('x1', domain)
        x2 = Variable('x2', domain)
        x3 = Variable('x3', domain)

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi(x1_, x2_, x3_):
            if x1_ == x2_:
                return 2
            if x1_ == x3_:
                return 1
            return 0

        g = GdbaComputation(x1, [phi], increase_mode='T', comp_def=MagicMock())
        c, _, _ = g.__constraints__[0]
        g._increase_cost(c)
        modifiers = g.__constraints_modifiers__[c]
        for _, modifier in modifiers.items():
            self.assertEqual(modifier, 1)
