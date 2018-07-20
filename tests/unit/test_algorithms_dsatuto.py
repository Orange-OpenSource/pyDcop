from pydcop.algorithms.objects import load_algorithm_module, ComputationDef, \
    AlgoDef
from pydcop.computations_graph.constraints_hypergraph import \
    VariableComputationNode
from pydcop.dcop.objects import Variable


def test_memory_footprint():

    dsa_module = load_algorithm_module('dsatuto')

    v1 = Variable('v1', [1,2])
    comp_def = ComputationDef(VariableComputationNode(v1, []),
                              AlgoDef('dsatuto'))
    comp = dsa_module.DsaTutoComputation(v1, [], comp_def)

    assert comp.footprint() == 1
