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

import pytest

from pydcop.dcop.objects import VariableWithCostDict
from pydcop.dcop.scenario import EventAction, Scenario, DcopEvent
from pydcop.dcop.yamldcop import (
    load_dcop,
    DcopInvalidFormatError,
    load_scenario,
    yaml_scenario,
)


def test_load_name_and_description():
    dcop = load_dcop(
        """
    name: dcop name
    description: dcop description
    objective: min
    """
    )

    assert dcop.name == "dcop name"
    assert dcop.description, "dcop description"
    assert dcop.objective, "min"


def test_raises_when_no_name():

    with pytest.raises(ValueError):
        load_dcop(
            """
    description: dcop description
    objective: max
    """
        )


def test_load_name_without_desc():

    dcop = load_dcop(
        """
    name: dcop name
    objective: max
    """
    )

    assert dcop.name == "dcop name"
    assert dcop.objective == "max"
    assert dcop.description == ""


def test_load_name_long_desc():

    dcop = load_dcop(
        """
    name: dcop name
    description: A long dcop description that span on several lines. Lorem 
                 ipsum sed dolores et tutti quanti.
    objective: max
    """
    )
    assert dcop.name == "dcop name"
    assert (
        dcop.description == "A long dcop description that span on several "
        "lines. Lorem ipsum sed dolores et tutti "
        "quanti."
    )


def test_raises_when_invalid_objective():

    with pytest.raises(ValueError):
        load_dcop(
            """
    name: dcop name
    description: dcop description
    objective: foo
    """
        )


def test_raises_when_no_objective():

    with pytest.raises(ValueError):
        load_dcop(
            """
    name: dcop name
    description: dcop description
    """
        )


class TestDcopLoadDomains(unittest.TestCase):
    def setUp(self):
        self.dcop_str = """
        name: dcop name
        description: dcop description
        objective: min        
        """

    def test_simple_domain(self):
        self.dcop_str += """
        domains:
          d1:
            type: d1_type
            values: [0, 1, 2]
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.domains), 1)
        d = dcop.domains["d1"]
        self.assertEqual(d.name, "d1")
        self.assertEqual(d.type, "d1_type")

    def test_extensive_int_domain(self):
        self.dcop_str += """
        domains:
          d1:
            type: d1_type
            values: [0, 1, 2]
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.domains), 1)
        d = dcop.domains["d1"]
        self.assertEqual(d.name, "d1")
        self.assertEqual(d.type, "d1_type")
        self.assertEqual(len(d.values), 3)
        self.assertEqual(d.values, (0, 1, 2))

    def test_range_int_domain(self):
        self.dcop_str += """
        domains:
          d1:
            type: d1_type
            values: [0 .. 10]
        """
        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.domains), 1)
        d = dcop.domains["d1"]
        self.assertEqual(d.name, "d1")
        self.assertEqual(d.type, "d1_type")
        self.assertEqual(len(d.values), 11)
        self.assertEqual(d.values, tuple(range(11)))

    def test_string_domain(self):
        self.dcop_str += """
        domains:
          d1:
            type: d1_type
            values: [A, B, C]
        """
        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.domains), 1)
        d = dcop.domains["d1"]
        self.assertEqual(d.name, "d1")
        self.assertEqual(d.type, "d1_type")
        self.assertEqual(len(d.values), 3)
        self.assertEqual(d.values, ("A", "B", "C"))

    def test_boolean_domain(self):
        self.dcop_str += """
        domains:
          d1_bool:
            type: d1_type
            values: [true, false]
        """
        dcop = load_dcop(self.dcop_str)

        d = dcop.domains["d1_bool"]
        self.assertEqual(d.name, "d1_bool")
        self.assertEqual(d.type, "d1_type")
        self.assertEqual(len(d.values), 2)
        self.assertEqual(d.values, (True, False))

    def test_several_domain(self):
        self.dcop_str += """
        domains:
          d_str:
            type: d1_type
            values: [A, B, C]
          d_bool:
            type: d1_type
            values: [true, false]
          d_range:
            type: d1_type
            values: [0 .. 10]
        """
        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.domains), 3)
        self.assertEqual(dcop.domains["d_str"].name, "d_str")
        self.assertEqual(len(dcop.domains["d_str"].values), 3)
        self.assertEqual(dcop.domains["d_bool"].name, "d_bool")
        self.assertEqual(len(dcop.domains["d_bool"].values), 2)
        self.assertEqual(dcop.domains["d_range"].name, "d_range")
        self.assertEqual(len(dcop.domains["d_range"].values), 11)


class TestDcopLoadVariables(unittest.TestCase):
    def setUp(self):
        self.dcop_str = """
        name: dcop name
        description: dcop description
        objective: min
        
        domains:
          d1:
            type: d1_type
            values: [0, 1, 2]                
        """

    def test_variable(self):

        self.dcop_str += """
        variables:
          v1:
            domain : d1     
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.variables), 1)
        self.assertIn("v1", dcop.variables)
        v1 = dcop.variable("v1")
        self.assertEqual(v1.name, "v1")
        self.assertEqual(v1.domain.name, "d1")

    def test_variable_with_initial_value(self):
        self.dcop_str += """
        variables:
          v1:
            domain : d1
            initial_value: 1
        """

        dcop = load_dcop(self.dcop_str)

        v1 = dcop.variable("v1")
        self.assertEqual(v1.initial_value, 1)

    def test_raise_when_invalid_initial_value(self):
        self.dcop_str += """
        variables:
          v1:
            domain : d1
            initial_value: N
        """

        self.assertRaises(ValueError, load_dcop, self.dcop_str)

    def test_several_variables(self):

        self.dcop_str += """
        variables:
          v1:
            domain : d1     
          v2:
            domain : d1     
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.variables), 2)
        self.assertIn("v1", dcop.variables)
        v1 = dcop.variable("v1")
        self.assertEqual(v1.name, "v1")
        self.assertEqual(v1.domain.name, "d1")

        self.assertIn("v2", dcop.variables)
        v2 = dcop.variable("v2")
        self.assertEqual(v2.name, "v2")
        self.assertEqual(v2.domain.name, "d1")


class TestDcopLoadVariablesWithCost(unittest.TestCase):
    def setUp(self):
        self.dcop_str = """
        name: dcop name
        description: dcop description
        objective: min

        domains:
          d1:
            type: d1_type
            values: [0 .. 10]                
        """

    def test_variable_with_cost_function(self):
        self.dcop_str += """
        variables:
          v1:
            domain : d1     
            cost_function: 0.2 * v1
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.variables), 1)
        self.assertIn("v1", dcop.variables)
        v1: VariableWithCostDict = dcop.variable("v1")
        self.assertEqual(v1.cost_for_val(1), 0.2)

    def test_variable_with_cost(self):
        self.dcop_str += """
          colors:
            type: color
            values: [R, G]
        
        variables:
          v1:
            domain: colors
            cost_function: " -0.1 if v1 == 'R' else 0.2"
          v2:
            domain: colors
            cost_function: "-0.4 if v2 == 'G' else 0.5"
        """
        dcop = load_dcop(self.dcop_str)
        v1: VariableWithCostDict = dcop.variable("v1")
        self.assertEqual(v1.cost_for_val("R"), -0.1)
        self.assertEqual(v1.cost_for_val("G"), 0.2)
        v2: VariableWithCostDict = dcop.variable("v2")
        self.assertEqual(v2.cost_for_val("R"), 0.5)
        self.assertEqual(v2.cost_for_val("G"), -0.4)

    def test_variable_with_noisy_cost_function(self):
        self.dcop_str += """
        variables:
          v1:
            domain : d1     
            cost_function: 0.2 * v1
            noise_level: 0.05
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.variables), 1)
        self.assertIn("v1", dcop.variables)
        v1: VariableWithCostDict = dcop.variable("v1")

        all_equal = True
        for v in dcop.domain("d1").values:
            noisy_cost = v1.cost_for_val(v)
            self.assertGreaterEqual(v * 0.2 + 0.05, noisy_cost)
            self.assertLessEqual(v * 0.2 - 0.05, v * 0.2)
            all_equal = all_equal and (noisy_cost == v * 0.2)

        # if all cost where exactly equal to the cost function, it means we
        # probably did not apply the noise
        self.assertFalse(all_equal)


class TestDcopLoadExternalVariables(unittest.TestCase):
    def setUp(self):
        self.dcop_str = """
        name: dcop name
        description: dcop description
        objective: min

        domains:
          d1:
            type: d1_type
            values: [0, 1, 2]                
        """

    def test_variable(self):

        self.dcop_str += """
        external_variables:
          ext_var1:
            domain: d1
            initial_value: 2
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.external_variables), 1)
        self.assertIn("ext_var1", dcop.external_variables)
        e1 = dcop.get_external_variable("ext_var1")
        self.assertEqual(e1.name, "ext_var1")
        self.assertEqual(e1.domain.name, "d1")


class TestDcopLoadConstraints(unittest.TestCase):
    def setUp(self):
        self.dcop_str = """
        name: dcop name
        description: dcop description
        objective: min

        domains:
          d1:
            type: d1_type
            values: [0 .. 10]             
          dbool:
            type: bool
            values: [true, false]
        variables:
          v1:
            domain : d1     
          v2:
            domain : d1     
        external_variables:
          e1:
            domain: dbool
            initial_value: true            
        """

    def test_one_var_constraint(self):

        self.dcop_str += """
        constraints:
          cost_v1:
            type: intention
            function: v1 * 0.4
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.constraints), 1)
        c = dcop.constraint("cost_v1")
        self.assertEqual(c.name, "cost_v1")
        self.assertEqual(c(v1=10), 4)

    def test_two_var_constraint(self):

        self.dcop_str += """
        constraints:
          ws:
            type: intention
            function: v1 * 4 - 2 * v2
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.constraints), 1)
        c = dcop.constraint("ws")
        self.assertEqual(c.name, "ws")
        self.assertEqual(c(v1=10, v2=3), 34)

    def test_external_var_constraint(self):

        self.dcop_str += """
        constraints:
          cond:
            type: intention
            function: 10 if e1 else 0
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.constraints), 1)
        c = dcop.constraint("cond")
        self.assertEqual(c.name, "cond")
        self.assertEqual(c(e1=True), 10)
        self.assertEqual(c(e1=False), 0)

    def test_complex_constraint(self):
        self.dcop_str += """
        constraints:
          cond:
            type: intention
            function: 10 if e1 and v1 == v2 else 0
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.constraints), 1)
        c = dcop.constraint("cond")
        self.assertEqual(c.name, "cond")
        self.assertEqual(c(e1=True, v1=1, v2=1), 10)
        self.assertEqual(c(e1=True, v1=1, v2=2), 0)
        self.assertEqual(c(e1=False, v1=1, v2=1), 0)
        self.assertEqual(c(e1=False, v1=1, v2=2), 0)

    def test_multiline_intention_constraint(self):
        self.dcop_str += """
        constraints:
          cond:
            type: intention
            function: | 
              if e1:
                  b = v1 * 2
              else:
                  b = 0
              return b + v2
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.constraints), 1)
        c = dcop.constraint("cond")
        self.assertEqual(c.name, "cond")
        self.assertEqual(c(e1=True, v1=1, v2=1), 3)
        self.assertEqual(c(e1=True, v1=1, v2=2), 4)
        self.assertEqual(c(e1=False, v1=1, v2=1), 1)
        self.assertEqual(c(e1=False, v1=1, v2=2), 2)

    def test_extensional_constraint_one_var(self):
        self.dcop_str += """
        constraints:
          cost_v1:
            type: extensional
            default: 10
            variables: v1
            values: 
                2: 2
                3: 5
                4: 0 | 3
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.constraints), 1)
        c = dcop.constraint("cost_v1")
        self.assertEqual(c.name, "cost_v1")
        self.assertEqual(len(c.dimensions), 1)
        self.assertEqual(c.dimensions[0].name, "v1")
        self.assertEqual(c(v1=2), 2)
        self.assertEqual(c(v1=5), 3)
        self.assertEqual(c(v1=0), 4)
        self.assertEqual(c(v1=3), 4)
        self.assertEqual(c(v1=7), 10)  # Default value

    def test_extensional_constraint_two_var(self):
        self.dcop_str += """
        constraints:
          ext_test:
            type: extensional
            default: 10
            variables: [v1, v2]
            values: 
                2: 2 2
                3: 5 5
                4: 1 1 | 3 3
                5: 1 3 | 2 3 | 3 4
                6: 3 1 | 3 2 | 4 3
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.constraints), 1)
        c = dcop.constraint("ext_test")
        self.assertEqual(c.name, "ext_test")
        self.assertEqual(len(c.dimensions), 2)
        self.assertEqual([v.name for v in c.dimensions], ["v1", "v2"])
        self.assertEqual(c(v1=2, v2=2), 2)
        self.assertEqual(c(v1=5, v2=5), 3)

        self.assertEqual(c(v1=1, v2=1), 4)
        self.assertEqual(c(v1=3, v2=3), 4)

        self.assertEqual(c(v1=1, v2=3), 5)
        self.assertEqual(c(v1=2, v2=3), 5)
        self.assertEqual(c(v1=3, v2=4), 5)

        self.assertEqual(c(v1=3, v2=1), 6)
        self.assertEqual(c(v1=3, v2=2), 6)
        self.assertEqual(c(v1=4, v2=3), 6)

        self.assertEqual(c(v1=7, v2=7), 10)  # Default value


class TestDcopLoadAgents(unittest.TestCase):
    def setUp(self):
        self.dcop_str = """
        name: dcop name
        description: dcop description
        objective: min

        domains:
          d1:
            type: d1_type
            values: [0 .. 10]             
          dbool:
            type: bool
            values: [true, false]
        variables:
          v1:
            domain : d1     
          v2:
            domain : d1     
        external_variables:
          e1:
            domain: dbool
            initial_value: true            
        """

    def test_one_agent(self):

        self.dcop_str += """
        agents:
            a1:
              capacity: 100
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.agents), 1)
        self.assertIn("a1", dcop.agents)
        a1 = dcop.agent("a1")
        self.assertEqual(a1.name, "a1")
        self.assertEqual(a1.capacity, 100)

    def test_one_agent_with_arbitrary_attr(self):

        self.dcop_str += """
        agents:
            a1:
              capacity: 100
              foo: 12
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.agents), 1)
        self.assertIn("a1", dcop.agents)
        a1 = dcop.agent("a1")
        self.assertEqual(a1.name, "a1")
        self.assertEqual(a1.capacity, 100)
        self.assertEqual(a1.foo, 12)

    def test_agents_with_default_route(self):

        self.dcop_str += """
        agents:
            a1:
              capacity: 100
            a2:
              capacity: 100
            a3:
              capacity: 100
        routes: 
            default: 42
            a1:
              a2: 10
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.agents), 3)
        self.assertIn("a1", dcop.agents)
        self.assertIn("a2", dcop.agents)

        self.assertEqual(dcop.agent("a2").default_route, 42)
        # route given in yaml str:
        self.assertEqual(dcop.agent("a1").route("a2"), 10)
        self.assertEqual(dcop.agent("a2").route("a1"), 10)
        # internal route : 0
        self.assertEqual(dcop.agent("a1").route("a1"), 0)
        self.assertEqual(dcop.agent("a2").route("a2"), 0)
        # route not given in str : defaults to default_route
        self.assertEqual(dcop.agent("a1").route("a3"), 42)
        self.assertEqual(dcop.agent("a2").route("a3"), 42)

    def test_agents_no_route_def(self):

        self.dcop_str += """
        agents:
            a1:
              capacity: 100
            a2:
              capacity: 100
            a3:
              capacity: 100
        """

        dcop = load_dcop(self.dcop_str)

        # no defaul given
        self.assertEqual(dcop.agent("a1").route("a3"), 1)

    def test_agents_no_default_route(self):

        self.dcop_str += """
        agents:
            a1:
              capacity: 100
            a2:
              capacity: 100
            a3:
              capacity: 100
        routes: 
            a1:
              a2: 10
        """

        dcop = load_dcop(self.dcop_str)

        self.assertEqual(len(dcop.agents), 3)
        self.assertIn("a1", dcop.agents)
        self.assertIn("a2", dcop.agents)

        # no defaul given
        # self.assertEqual(dcop.route('a1', 'a3'), 1)
        self.assertEqual(dcop.agent("a1").route("a3"), 1)

    def test_duplicate_route_raises(self):
        self.dcop_str += """
        agents:
            a1:
              capacity: 100
            a2:
              capacity: 100
            a3:
              capacity: 100
        routes: 
            a1:
              a2: 10
            a2:
              a1: 5
        """
        self.assertRaises(DcopInvalidFormatError, load_dcop, self.dcop_str)

    def test_default_global_hosting_costs(self):
        self.dcop_str += """
        agents:
            a1:
              capacity: 100
            a2:
              capacity: 100
            a3:
              capacity: 100
        hosting_costs: 
            default: 5
        """
        dcop = load_dcop(self.dcop_str)

        self.assertEqual(dcop.agent("a1").hosting_cost("foo"), 5)
        self.assertEqual(dcop.agent("a2").hosting_cost("bar"), 5)
        self.assertEqual(dcop.agent("a3").hosting_cost("foo"), 5)

    def test_default_agt_hosting_costs(self):
        self.dcop_str += """
        agents:
            a1:
              capacity: 100
            a2:
              capacity: 100
            a3:
              capacity: 100
        hosting_costs: 
            default: 5
            a1: 
              default: 3
        """
        dcop = load_dcop(self.dcop_str)

        self.assertEqual(dcop.agent("a1").hosting_cost("foo"), 3)
        self.assertEqual(dcop.agent("a1").hosting_cost("bar"), 3)
        self.assertEqual(dcop.agent("a2").hosting_cost("bar"), 5)
        self.assertEqual(dcop.agent("a3").hosting_cost("foo"), 5)

    def test_comp_agt_hosting_costs(self):
        self.dcop_str += """
        agents:
            a1:
              capacity: 100
            a2:
              capacity: 100
            a3:
              capacity: 100
        hosting_costs: 
            default: 5
            a1: 
              default: 3
              computations:
                foo: 7
        """
        dcop = load_dcop(self.dcop_str)

        self.assertEqual(dcop.agent("a1").hosting_cost("foo"), 7)
        self.assertEqual(dcop.agent("a1").hosting_cost("bar"), 3)
        self.assertEqual(dcop.agent("a2").hosting_cost("bar"), 5)
        self.assertEqual(dcop.agent("a3").hosting_cost("foo"), 5)


class TestLoadDistributionHintsMustHost(unittest.TestCase):
    def setUp(self):
        self.dcop_str = """
        name: dcop name
        description: dcop description
        objective: min

        domains:
          d1:
            type: d1_type
            values: [0 .. 10]             
          dbool:
            type: bool
            values: [true, false]
        variables:
          v1:
            domain : d1     
          v2:
            domain : d1
        constraints:
          c1:
            type: intention
            function: v1 * 0.4                 
        external_variables:
          e1:
            domain: dbool
            initial_value: true
        agents:
            a1:
              capacity: 100                        
        """

    def test_no_dist_hint(self):

        dcop = load_dcop(self.dcop_str)
        self.assertIsNone(dcop.dist_hints)

    def test_must_host(self):
        self.dcop_str += """
        distribution_hints:
            must_host:
              a1: [v1]
        """
        dcop = load_dcop(self.dcop_str)
        hints = dcop.dist_hints
        self.assertIsNotNone(hints)

        self.assertIn("v1", hints.must_host("a1"))

    def test_must_host_several(self):
        self.dcop_str += """
        distribution_hints:
            must_host:
              a1: [v1, c1]
        """
        dcop = load_dcop(self.dcop_str)
        hints = dcop.dist_hints
        self.assertIsNotNone(hints)

        self.assertIn("v1", hints.must_host("a1"))
        self.assertIn("c1", hints.must_host("a1"))

    def test_must_host_returns_empty(self):
        self.dcop_str += """
        distribution_hints:
            must_host:
              a1: [v1]
        """
        dcop = load_dcop(self.dcop_str)
        hints = dcop.dist_hints
        self.assertIsNotNone(hints)

        self.assertEqual(len(hints.must_host("a5")), 0)

    def test_raises_on_invalid_agent_in_must_host(self):
        self.dcop_str += """
        distribution_hints:
            must_host:
              a10: [v1]
        """
        self.assertRaises(ValueError, load_dcop, self.dcop_str)

    def test_raises_on_invalid_var_in_must_host(self):
        self.dcop_str += """
        distribution_hints:
            must_host:
              a1: [f1]
        """
        self.assertRaises(ValueError, load_dcop, self.dcop_str)


class TestLoadDistributionHintsHostWith(unittest.TestCase):
    def setUp(self):
        self.dcop_str = """
        name: dcop name
        description: dcop description
        objective: min

        domains:
          d1:
            type: d1_type
            values: [0 .. 10]             
          dbool:
            type: bool
            values: [true, false]
        variables:
          v1:
            domain : d1     
          v2:
            domain : d1
        constraints:
          c1:
            type: intention
            function: v1 * 0.4                 
        external_variables:
          e1:
            domain: dbool
            initial_value: true
        agents:
            a1:
              capacity: 100                        
        """

    def test_no_hints(self):
        dcop = load_dcop(self.dcop_str)
        hints = dcop.dist_hints

        self.assertIsNone(hints)

    def test_host_with(self):
        self.dcop_str += """
        distribution_hints:
            host_with:
              v1: [c1]
        """
        dcop = load_dcop(self.dcop_str)
        hints = dcop.dist_hints

        self.assertIn("c1", hints.host_with("v1"))
        self.assertIn("v1", hints.host_with("c1"))

    def test_host_with_several(self):
        self.dcop_str += """
        distribution_hints:
            host_with:
              v1: [c1, v2]
        """
        dcop = load_dcop(self.dcop_str)
        hints = dcop.dist_hints

        self.assertIn("c1", hints.host_with("v1"))
        self.assertIn("c1", hints.host_with("v2"))
        self.assertIn("v1", hints.host_with("c1"))
        self.assertIn("v1", hints.host_with("v2"))
        self.assertIn("v2", hints.host_with("v1"))
        self.assertIn("v2", hints.host_with("c1"))


def test_load_scenario():
    s = """
inputs:
  origin: hand-made
events:
  - id: w1
    delay: 30
  - id: e1
    actions:
      - type: remove_agent
        agent: a005
      - type: remove_agent
        agent: a068
  - id: w2
    delay: 10
  - id: e2
    actions:
      - type: remove_agent
        agent: a026
      - type: remove_agent
        agent: a056
    """

    scenario = load_scenario(s)

    assert len(scenario.events) == 4
    assert scenario.events[0].is_delay
    assert len(scenario.events[1].actions) == 2
    assert scenario.events[3].actions[1].type == "remove_agent"
    assert scenario.events[3].actions[0].args["agent"] == "a026"


def test_yaml_scenario_one_event():
    events = [DcopEvent("1", actions=[EventAction("remove_agent", agent="a01")])]
    scenario = Scenario(events)

    scenario_str = yaml_scenario(scenario)

    obtained = load_scenario(scenario_str)

    assert len(obtained.events) == 1
    assert len(obtained.events[0].actions) == 1


def test_yaml_scenario_one_event_two_actions():
    events = [
        DcopEvent(
            "1",
            actions=[
                EventAction("remove_agent", agent="a01"),
                EventAction("remove_agent", agent="a05"),
            ],
        )
    ]
    scenario = Scenario(events)

    scenario_str = yaml_scenario(scenario)

    obtained = load_scenario(scenario_str)

    assert len(obtained.events) == 1
    assert len(obtained.events[0].actions) == 2
    assert obtained.events[0].actions[1].type == "remove_agent"
    assert obtained.events[0].actions[1].args["agent"] == "a05"


def test_yaml_scenario_two_events():
    events = [
        DcopEvent(
            "1",
            actions=[
                EventAction("remove_agent", agent="a01"),
                EventAction("remove_agent", agent="a05"),
            ],
        ),
        DcopEvent("2", delay=30),
    ]
    scenario = Scenario(events)

    scenario_str = yaml_scenario(scenario)

    obtained = load_scenario(scenario_str)

    assert len(obtained.events) == 2
    assert obtained.events[1].is_delay
    assert not obtained.events[0].is_delay
