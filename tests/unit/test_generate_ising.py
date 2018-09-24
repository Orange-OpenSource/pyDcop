import networkx as nx

from pydcop.commands.generators.ising import (
    generate_unary_extensive_constraint,
    generate_unary_intentional_constraint,
    generate_unary_constraints,
    generate_binary_intentional_constraint,
    generate_binary_variables,
    generate_binary_constraints, generate_binary_extensive_constraint)
from pydcop.dcop.objects import Variable, Domain
from pydcop.dcop.relations import NAryMatrixRelation, Constraint, NAryFunctionRelation


def test_generate_unary_constraints():
    domain = Domain("d", "d", [0, 1])
    variables = {str(i): Variable(f"v_{i}", domain) for i in range(10)}
    k = 0.05

    constraints = generate_unary_constraints(variables, k, extensive=True)

    assert len(constraints) == 10
    for constraint in constraints.values():
        assert isinstance(constraint, Constraint)
        assert -k <= constraint(0) <= k
        assert -k <= constraint(1) <= k
        assert -constraint(1) == constraint(0)


def test_generate_unary_extensive_constraint():
    domain = Domain("d", "d", [0, 1])
    variable = Variable("v", domain)
    k = 0.05

    constraint = generate_unary_extensive_constraint(variable, k)

    assert isinstance(constraint, Constraint)
    assert type(constraint) == NAryMatrixRelation
    assert -k <= constraint(0) <= k
    assert -k <= constraint(1) <= k
    assert -constraint(1) == constraint(0)


def test_generate_unary_intentional_constraint():
    domain = Domain("d", "d", [0, 1])
    variable = Variable("v", domain)
    k = 0.05

    constraint = generate_unary_intentional_constraint(variable, k)

    assert isinstance(constraint, Constraint)
    assert type(constraint) == NAryFunctionRelation
    assert -k <= constraint(0) <= k
    assert -k <= constraint(1) <= k
    assert -constraint(1) == constraint(0)


def test_generate_binary_intentional_constraint():
    domain = Domain("d", "d", [0, 1])
    variable1 = Variable("v1", domain)
    variable2 = Variable("v2", domain)
    b = 1.6

    constraint = generate_binary_intentional_constraint(variable1, variable2, b)

    assert type(constraint) == NAryFunctionRelation
    check_binary_constraint(constraint, bin_range=b)

def test_generate_binary_extensive_constraint():
    domain = Domain("d", "d", [0, 1])
    variable1 = Variable("v1", domain)
    variable2 = Variable("v2", domain)
    b = 1.6

    constraint = generate_binary_extensive_constraint(variable1, variable2, b)

    assert type(constraint) == NAryMatrixRelation
    check_binary_constraint(constraint, bin_range=b)


def test_generate_binary_variables():
    grid_graph = nx.grid_2d_graph(4, 5, periodic=True)
    domain = Domain("d", "d", [0, 1])

    variables = generate_binary_variables(grid_graph, domain)

    assert len(variables) == 4 * 5
    for name, variable in variables.items():
        assert variable.domain == domain
        assert name == variable.name


def test_generate_binary_constraints():
    row_count, col_count = 3, 3
    grid_graph = nx.grid_2d_graph(row_count, col_count, periodic=True)
    domain = Domain("d", "d", [0, 1])
    variables = generate_binary_variables(grid_graph, domain)
    bin_range= 1.6

    constraints = generate_binary_constraints(grid_graph, variables, bin_range, True)
    assert len(constraints) == len(list(grid_graph.edges))
    for constraint in constraints.values():
        assert type(constraint) == NAryMatrixRelation
        check_binary_constraint(constraint, bin_range)

    constraints = generate_binary_constraints(grid_graph, variables, bin_range, False)
    assert len(constraints) == len(list(grid_graph.edges))
    for constraint in constraints.values():
        assert type(constraint) == NAryFunctionRelation
        check_binary_constraint(constraint, bin_range)


def check_binary_constraint(constraint, bin_range):
    v1, v2 = constraint.dimensions
    assert isinstance(constraint, Constraint)
    value = constraint(**{v1.name:0, v2.name:0})
    assert -bin_range < value < bin_range
    assert constraint(**{v1.name:0, v2.name:0}) == value
    assert constraint(**{v1.name:1, v2.name:1}) == value
    assert constraint(**{v1.name:0, v2.name:1}) == -value
    assert constraint(**{v1.name:1, v2.name:0}) == -value
