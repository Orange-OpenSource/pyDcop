from pydcop.commands.generators.agents import (
    find_corresponding_variables,
    generate_agents_names,
    generate_hosting_costs,
    generate_agents_from_variables,
    find_prefix,
    generate_agents_from_count,
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


def test_find_vars_agts_mapping_dual_naming():
    # Test when name is using a scheme with two numbers
    agents = ['a0_0', 'a0_1', 'a0_2', 'a0_3', 'a0_4', 'a0_5', 'a1_0', 'a1_1', 'a1_2', 'a1_3', 'a1_4', 'a1_5', 'a2_0', 'a2_1', 'a2_2', 'a2_3', 'a2_4', 'a2_5', 'a3_0', 'a3_1', 'a3_2', 'a3_3', 'a3_4', 'a3_5', 'a4_0', 'a4_1', 'a4_2', 'a4_3', 'a4_4', 'a4_5', 'a5_0', 'a5_1', 'a5_2', 'a5_3', 'a5_4', 'a5_5']
    variables =  ['v_0_0', 'v_0_1', 'v_0_2', 'v_0_3', 'v_0_4', 'v_0_5', 'v_1_0', 'v_1_1', 'v_1_2', 'v_1_3', 'v_1_4', 'v_1_5', 'v_2_0', 'v_2_1', 'v_2_2', 'v_2_3', 'v_2_4', 'v_2_5', 'v_3_0', 'v_3_1', 'v_3_2', 'v_3_3', 'v_3_4', 'v_3_5', 'v_4_0', 'v_4_1', 'v_4_2', 'v_4_3', 'v_4_4', 'v_4_5', 'v_5_0', 'v_5_1', 'v_5_2', 'v_5_3', 'v_5_4', 'v_5_5']
    obtained = find_corresponding_variables(agents, variables)

    assert obtained["a0_0"] == "v_0_0"
    assert obtained["a0_1"] == "v_0_1"


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

    agents = generate_agents_from_count(10)
    assert len(agents) == 10
    agents = {a for a in agents}
    assert len(agents) == 10
    assert "a5" in agents
    assert "a7" in agents


def test_generate_10_agents_with_prefix():

    agents = generate_agents_from_count(10, agent_prefix="agt_")
    assert len(agents) == 10
    agents = {a for a in agents}
    assert len(agents) == 10
    assert "agt_5" in agents
    assert "agt_7" in agents


def test_generate_100_agents():

    agents = generate_agents_from_count(100)
    assert len(agents) == 100
    agents = {a for a in agents}
    # Names must be unique
    assert len(agents) == 100

    assert "a05" in agents
    assert "a07" in agents


def test_generate_hosting_costs():
    agents = generate_agents_from_count(10)
    variables = create_variables("v", range(20), Domain("color", "", [1, 2]))

    costs = generate_hosting_costs("name_mapping", agents, list(variables))

    assert costs["a1"]["v01"] == 0
    assert costs["a9"]["v09"] == 0
    assert costs["a0"]["v00"] == 0

    mapped_vars = [costs.values()]
    assert "v12" not in mapped_vars


def test_find_prefix_len1():
    obtained = find_prefix(["x1", "x2", "x3"])
    assert obtained == "x"


def test_find_prefix_no_common_prefix():
    obtained = find_prefix(["x1", "x2", "V3"])
    assert obtained == ""


def test_find_prefix_len2():
    obtained = find_prefix(["x_1", "x_2", "x_3"])
    assert obtained == "x_"


def test_generate_agent_from_variables():

    variables = ["v1", "v2", "v3", "v4"]
    obtained = generate_agents_from_variables(variables)
    assert len(obtained) == 4
    assert "a1" in obtained
    assert "a2" in obtained
    assert "a3" in obtained
    assert "a4" in obtained


def test_generate_agent_from_variables_with_prefix():

    variables = ["v1", "v2", "v3", "v4"]
    obtained = generate_agents_from_variables(variables, agent_prefix="A_")
    assert len(obtained) == 4
    assert "A_1" in obtained
    assert "A_2" in obtained
    assert "A_3" in obtained
    assert "A_4" in obtained
