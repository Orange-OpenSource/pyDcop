from pydcop.commands.generators.agents import (
    find_corresponding_variables,
    generate_agents_names,
    generate_hosting_costs,
)
from pydcop.dcop.objects import create_variables, Domain


def test_find_vars_agts_mapping():
    agents = ["a1", "a2", "a3"]
    variables = ["v1", "v2", "v3"]

    obtained = find_corresponding_variables(agents, variables)

    assert obtained["a1"] == "v1"
    assert obtained["a2"] == "v2"
    assert obtained["a3"] == "v3"


def test_find_vars_agts_mapping_different_padding():
    agents = ["a1", "a2", "a3"]
    variables = ["v01", "v02", "v03"]

    obtained = find_corresponding_variables(agents, variables)

    assert obtained["a1"] == "v01"
    assert obtained["a2"] == "v02"
    assert obtained["a3"] == "v03"


def test_find_vars_agts_mapping_with_prefix():
    agents = ["agt_1", "agt_2", "agt_3"]
    variables = ["Var_1", "Var_2", "Var_3"]

    obtained = find_corresponding_variables(
        agents, variables, agt_prefix="agt_", var_prefix="Var_"
    )

    assert obtained["agt_1"] == "Var_1"
    assert obtained["agt_2"] == "Var_2"
    assert obtained["agt_3"] == "Var_3"


def test_generate_10_agents():

    agents = generate_agents_names(10)
    assert len(agents) == 10
    agents = {a for a in agents}
    assert len(agents) == 10
    assert "a5" in agents
    assert "a7" in agents


def test_generate_10_agents_with_prefix():

    agents = generate_agents_names(10, agent_prefix="agt_")
    assert len(agents) == 10
    agents = {a for a in agents}
    assert len(agents) == 10
    assert "agt_5" in agents
    assert "agt_7" in agents


def test_generate_100_agents():

    agents = generate_agents_names(100)
    assert len(agents) == 100
    agents = {a for a in agents}
    # Names must be unique
    assert len(agents) == 100

    assert "a05" in agents
    assert "a07" in agents


def test_generate_hosting_costs():
    agents = generate_agents_names(10)
    variables = create_variables("v", range(20), Domain("color", "", [1, 2]))

    costs = generate_hosting_costs("name_mapping", agents, list(variables))

    assert costs["a1"]["v01"] == 0
    assert costs["a9"]["v09"] == 0
    assert costs["a0"]["v00"] == 0

    mapped_vars = [costs.values()]
    assert "v12" not in mapped_vars
