from pydcop.commands.generators.agents import (
    find_corresponding_variables,
    generate_agents_names,
    generate_hosting_costs,
    generate_agents_from_variables,
    find_prefix,
    generate_agents_from_count,
    find_corresponding_variables_start_with,
)
from pydcop.dcop.objects import create_variables, Domain


def test_find_vars_agts_mapping():
    agents = ["a1", "a2", "a3"]
    variables = ["v1", "v2", "v3"]

    obtained = find_corresponding_variables(agents, variables)

    assert "v1" in obtained["a1"]
    assert "v2" in obtained["a2"]
    assert "v3" in obtained["a3"]


def test_find_vars_agts_mapping_different_padding():
    agents = ["a1", "a2", "a3"]
    variables = ["v01", "v02", "v03"]

    obtained = find_corresponding_variables(agents, variables)

    assert "v01" in obtained["a1"]
    assert "v02" in obtained["a2"]
    assert "v03" in obtained["a3"]


def test_find_vars_agts_mapping_dual_naming():
    # Test when name is using a scheme with two numbers
    agents = ["a0_0", "a0_1", "a0_2", "a0_3", "a0_4", "a0_5", "a1_0"]
    variables = ["v_0_0", "v_0_1", "v_0_2", "v_0_3", "v_0_4", "v_0_5", "v_1_0", "v_1_1"]
    obtained = find_corresponding_variables(agents, variables)

    assert "v_0_0" in obtained["a0_0"]
    assert "v_0_1" in obtained["a0_1"]


def test_find_vars_agts_mapping_with_prefix():
    agents = ["agt_1", "agt_2", "agt_3"]
    variables = ["Var_1", "Var_2", "Var_3"]

    obtained = find_corresponding_variables(
        agents, variables, agt_prefix="agt_", var_prefix="Var_"
    )

    assert "Var_1" in obtained["agt_1"]
    assert "Var_2" in obtained["agt_2"]
    assert "Var_3" in obtained["agt_3"]


def test_find_vars_starts_with():
    agents = ["agt_1", "agt_2", "agt_3"]
    variables = ["Var_1_0", "Var_2_1", "Var_2_2", "Var_3_3"]

    obtained = find_corresponding_variables_start_with(agents, variables)

    assert "Var_1_0" in obtained["agt_1"]
    assert "Var_2_1" in obtained["agt_2"]
    assert "Var_2_2" in obtained["agt_2"]
    assert "Var_3_3" in obtained["agt_3"]


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

    mapping = {"a1": ["v01", "v02"], "a2": [], "a9": ["v09"], "a0": ["v00"]}

    costs = generate_hosting_costs("name_mapping", mapping)

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
