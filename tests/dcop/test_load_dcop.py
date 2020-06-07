"""

Tests for loading DCOP yaml files.

These tests check for correct parsing of the DCOP definition files.

"""
import unittest
from pydcop.dcop.yamldcop import load_dcop_from_file, load_dcop
from pydcop.dcop.objects import (
    ContinuousDomain,
    Domain,
)

dcop_test_str = """
    name: 'dcop test'
    description: 'Testing of DCOP yaml parsing'
    objective: min

    domains: 
        dint:
            values  : [0, 1, 2, 3, 4]
            type   : non_semantic
        dstr:
            values  : ['A', 'B', 'C', 'D', 'E']
        dcont:
            values  : [0 .. 1]
            type   : non_semantic
            initial_value: 3
        dbool:
            values  : [true, false]

    variables:
        var1:
            domain: dint
            initial_value: 0
            yourkey: yourvalue
            foo: bar
        var2:
            domain: dstr
            initial_value: 'A'
        var3:
            domain: dint
            initial_value: 0
            cost_function: var3 * 0.5
        var4:
            domain: dcont
            initial_value: 0
            cost_function: var4 * 0.6
    
    external_variables:
        ext_var1:
            domain: dbool
            initial_value: False

    constraints:
        c1:
            type: intention
            function: var3 - var1

    agents:
        a1:
            capacity: 100
        a2:
            capacity: 100
        a3:
            capacity: 100
        a4:
            capacity: 100
"""

class TestDomains(unittest.TestCase):

    def test_classes(self):
        # Load the DCOP
        dcop = load_dcop(dcop_test_str)
        
        # Check the classes
        self.assertIsInstance(dcop.domains['dint'], Domain)
        self.assertIsInstance(dcop.domains['dstr'], Domain)
        self.assertIsInstance(dcop.domains['dcont'], ContinuousDomain)
        self.assertIsInstance(dcop.domains['dbool'], Domain)

    def test_values(self):
        # Load the DCOP
        dcop = load_dcop(dcop_test_str)

        # Check the values
        self.assertEqual(dcop.domains['dint'].values, (0, 1, 2, 3, 4))
        self.assertEqual(dcop.domains['dstr'].values, ('A', 'B', 'C', 'D', 'E'))
        self.assertEqual(dcop.domains['dcont'].values, (0.0, 1.0))
        self.assertEqual(dcop.domains['dbool'].values, (True, False))

    def test_bounds(self):
        # Load the DCOP
        dcop = load_dcop(dcop_test_str)

        # Check the bounds
        self.assertEqual(dcop.domains['dcont'].lower_bound, 0.0)
        self.assertEqual(dcop.domains['dcont'].upper_bound, 1.0)
