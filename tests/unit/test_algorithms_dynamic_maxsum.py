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


import types
import unittest
from unittest.mock import MagicMock

from pydcop.algorithms.maxsum_dynamic import DynamicFunctionFactorComputation
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import AsNAryFunctionRelation
from pydcop.infrastructure.communication import InProcessCommunicationLayer, Messaging

#
class DynamicFunctionFactorComputationTest(unittest.TestCase):
    def test_init(self):

        domain = list(range(10))
        x1 = Variable("x1", domain)
        x2 = Variable("x2", domain)

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            return x1_ + x2_

        comp_def = MagicMock()
        comp_def.algo.algo = "amaxsum"
        comp_def.algo.mode = "min"
        comp_def.node.factor = phi
        f = DynamicFunctionFactorComputation(comp_def=comp_def)

        self.assertEqual(f.name, "phi")

    def test_change_function_name(self):
        domain = list(range(10))
        x1 = Variable("x1", domain)
        x2 = Variable("x2", domain)

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            return x1_ + x2_

        @AsNAryFunctionRelation(x1, x2)
        def phi2(x1_, x2_):
            return x1_ - x2_

        comp_def = MagicMock()
        comp_def.algo.algo = "amaxsum"
        comp_def.algo.mode = "min"
        comp_def.node.factor = phi
        f = DynamicFunctionFactorComputation(comp_def=comp_def)
        f.message_sender = MagicMock()
        f.change_factor_function(phi2)

        self.assertEqual(f.name, "phi")

    def test_change_function_different_order(self):
        domain = list(range(10))
        x1 = Variable("x1", domain)
        x2 = Variable("x2", domain)

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            return x1_ + x2_

        @AsNAryFunctionRelation(x2, x1)
        def phi2(x2_, x1_):
            return x1_ - x2_

        comp_def = MagicMock()
        comp_def.algo.algo = "amaxsum"
        comp_def.algo.mode = "min"
        comp_def.node.factor = phi
        f = DynamicFunctionFactorComputation(comp_def=comp_def)
        f.message_sender = MagicMock()
        f.change_factor_function(phi2)

        self.assertEqual(f.name, "phi")

    def test_change_function_wrong_dimensions_len(self):

        domain = list(range(10))
        x1 = Variable("x1", domain)
        x2 = Variable("x2", domain)
        x3 = Variable("x3", domain)

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            return x1_ + x2_

        @AsNAryFunctionRelation(x1, x2, x3)
        def phi2(x1_, x2_, x3_):
            return x1_ - x2_ + x3_

        comp_def = MagicMock()
        comp_def.algo.algo = "amaxsum"
        comp_def.algo.mode = "min"
        comp_def.node.factor = phi

        f = DynamicFunctionFactorComputation(comp_def=comp_def)
        # Monkey patch post_msg method with dummy mock to avoid error:
        f.post_msg = types.MethodType(lambda w, x, y, z: None, f)

        self.assertRaises(ValueError, f.change_factor_function, phi2)

    def test_change_function_wrong_dimensions_var(self):
        domain = list(range(10))
        x1 = Variable("x1", domain)
        x2 = Variable("x2", domain)
        x3 = Variable("x3", domain)

        @AsNAryFunctionRelation(x1, x2)
        def phi(x1_, x2_):
            return x1_ + x2_

        @AsNAryFunctionRelation(x1, x3)
        def phi2(x1_, x3_):
            return x1_ + x3_

        comp_def = MagicMock()
        comp_def.algo.algo = "amaxsum"
        comp_def.algo.mode = "min"
        comp_def.node.factor = phi

        f = DynamicFunctionFactorComputation(comp_def=comp_def)

        # Monkey patch post_msg method with dummy mock to avoid error:
        f.post_msg = types.MethodType(lambda w, x, y, z: None, f)

        self.assertRaises(ValueError, f.change_factor_function, phi2)
