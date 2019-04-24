import tempfile
from unittest.mock import call, patch
from unittest import mock

import yaml
import pydcop.commands.batch as batch_module

from pydcop.commands.batch import (
    parameters_configuration,
    regularize_parameters,
    run_batches,
    build_option_for_parameters,
    expand_variables,
    build_final_command,
    input_files_glob,
    input_files_re,
)


def test_input_files_glob(tmpdir):
    dir_path = str(tmpdir.realpath())

    tmpdir.join("dcop_ising_1.yaml").write("")
    tmpdir.join("dcop_ising_2.yaml").write("")
    tmpdir.join("dcop_coloring_3.yaml").write("")

    files = input_files_glob(f"{dir_path}/dcop_ising*.yaml")

    assert f"{dir_path}/dcop_ising_1.yaml" in files
    assert f"{dir_path}/dcop_ising_2.yaml" in files
    assert f"{dir_path}/dcop_coloring_3.yaml" not in files


def test_input_files_re(tmpdir):
    dir_path = str(tmpdir.realpath())

    tmpdir.join("dcop_ising_1.yaml").write("")
    tmpdir.join("dcop_ising_1_dist.yaml").write("")
    tmpdir.join("dcop_ising_2.yaml").write("")
    tmpdir.join("dcop_coloring_3.yaml").write("")
    tmpdir.join("dcop_coloring_3_dist.yaml").write("")

    files, extras, contexts = input_files_re(
        dir_path, "dcop_(?P<index>.*).yaml", ["dcop_{index}_dist.yaml"]
    )

    assert "dcop_ising_1.yaml" in files
    assert "dcop_ising_2.yaml" not in files
    assert "dcop_coloring_3.yaml" in files

    index = files.index("dcop_ising_1.yaml")
    assert "dcop_ising_1_dist.yaml" in extras[index]

    for context in contexts:
        assert "index" in context


def test_params_configuration_one_parameter():

    confs = parameters_configuration({"param1": ["v1_1", "v_1"]})

    assert len(confs) == 2  # two combinations
    for conf in confs:
        # One single param in each configuration
        assert len(conf) == 1
        assert "param1" in conf
        assert conf["param1"] in ["v1_1", "v_1"]


def test_params_configuration_two_parameters():

    p1_values = ["v1_1", "v1_2"]
    p2_values = ["v2_1"]

    confs = parameters_configuration({"p1": p1_values, "p2": p2_values})

    assert len(confs) == 2  # two combinations
    for conf in confs:
        # all parameters must be defined in each configuration
        assert len(conf) == 2
        assert "p1" in conf
        assert conf["p1"] in p1_values

        assert "p2" in conf
        assert conf["p2"] in p2_values


def test_params_configuration_two_parameters_2_3():

    p1_values = ["v1_1", "v1_2"]
    p2_values = ["v2_1", "v2_1", "v2_2"]

    confs = parameters_configuration({"p1": p1_values, "p2": p2_values})

    assert len(confs) == 6
    for conf in confs:
        # all parameters must be defined in each configuration
        assert len(conf) == 2
        assert "p1" in conf
        assert conf["p1"] in p1_values

        assert "p2" in conf
        assert conf["p2"] in p2_values


def test_params_configuration_two_parameters_with_dict():

    p1_values = ["v1_1", "v1_2"]
    p2_values = {"p21": ["a", "b"], "p22": ["c", "d"]}

    confs = parameters_configuration({"p1": p1_values, "p2": p2_values})

    assert len(confs) == 8
    for conf in confs:
        # all parameters must be defined in each configuration
        assert len(conf) == 2
        assert "p1" in conf
        assert conf["p1"] in p1_values

        assert "p2" in conf
        assert isinstance(conf["p2"], dict)
        assert conf["p2"]["p21"] in ["a", "b"]
        assert conf["p2"]["p22"] in ["c", "d"]


def test_params_configuration_list_with_single_element():
    confs = parameters_configuration({"p1": ["1"], "p2": ["2"]})

    assert len(confs) == 1
    assert confs[0]["p1"] == "1"
    assert confs[0]["p2"] == "2"


def test_params_configuration_order():
    conf1 = parameters_configuration({"b": ["1", "2"], "a": ["3", "4"]})
    conf2 = parameters_configuration({"b": ["2", "1"], "a": ["3", "4"]})
    conf3 = parameters_configuration({"a": ["3", "4"], "b": ["1", "2"]})
    conf4 = parameters_configuration({"a": ["4", "3"], "b": ["1", "2"]})

    assert conf1 == conf2
    assert conf1 == conf3
    assert conf1 == conf4


def test_params_configuration_order_with_dict():
    conf1 = parameters_configuration({"b": {"A": ["1", "2"]}, "a": ["3", "4"]})
    conf2 = parameters_configuration({"a": ["3", "4"], "b": {"A": ["1", "2"]}})
    conf3 = parameters_configuration({"a": ["4", "3"], "b": {"A": ["2", "1"]}})

    assert conf1 == conf2
    assert conf1 == conf3


def test_regularize_parameters():

    params_yaml = """
params:
  stop_cycle: 100
  variant: [A, B, C]
  probability: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
"""

    params = yaml.load(params_yaml, Loader=yaml.FullLoader)

    reg_params = regularize_parameters(params["params"])
    assert len(reg_params) == 3

    assert len(reg_params["stop_cycle"]) == 1
    for v in reg_params["stop_cycle"]:
        assert isinstance(v, str)

    assert len(reg_params["variant"]) == 3
    for v in reg_params["variant"]:
        assert isinstance(v, str)

    assert len(reg_params["probability"]) == 6
    for v in reg_params["probability"]:
        assert isinstance(v, str)


def test_regularize_parameters_imbricated():

    params_yaml = """
command_options:
    algo: [dsa, mgm]
    algo_params:
      stop_cycle: [50, 100]
      probability: [0.5, 0.6]
"""

    params = yaml.load(params_yaml, Loader=yaml.FullLoader)

    reg_params = regularize_parameters(params["command_options"])
    assert len(reg_params) == 2

    assert len(reg_params["algo"]) == 2
    for v in reg_params["algo"]:
        assert isinstance(v, str)

    assert len(reg_params["algo_params"]) == 2
    assert len(reg_params["algo_params"]["stop_cycle"]) == 2
    for v in reg_params["algo_params"]["stop_cycle"]:
        assert isinstance(v, str)

    assert len(reg_params["algo_params"]["probability"]) == 2
    for v in reg_params["algo_params"]["probability"]:
        assert isinstance(v, str)


def test_build_option_for_parameters():
    params = {"p1": "v1", "p2": "v2"}

    option_str = build_option_for_parameters(params)

    assert option_str == "--p1 v1 --p2 v2"


def test_build_option_for_parameters_with_dict_as_value():
    params = {"p1": "v1", "p2": {"p21": "v21", "p22": "v22"}}

    option_str = build_option_for_parameters(params)

    assert option_str == "--p1 v1 --p2 p21:v21 --p2 p22:v22"


def test_build_option_for_algo_parameters():
    params = {"p1": "v1", "p2": "v2"}

    option_str = build_option_for_parameters({"algo_params": params})

    assert option_str == "--algo_params p1:v1 --algo_params p2:v2"


def test_expand_variables_in_str():
    assert "abc" == expand_variables("a{v}c", {"v": "b"})
    assert "foo" == expand_variables("{bar}", {"bar": "foo"})


def test_expand_variables_in_str_with_subdir():
    context = {"d": {"a": "_insubdict_"}}
    obtained = expand_variables("a{d[a]}c", context)

    assert "a_insubdict_c" == obtained


def test_expand_variables_in_list():

    to_expand = ["abc", "expanded_{foo}", "{foo2}_expanded"]

    obtained = expand_variables(to_expand, {"foo": "bar", "foo2": "bar2"})

    assert obtained[0] == "abc"
    assert obtained[1] == "expanded_bar"
    assert obtained[2] == "bar2_expanded"


def test_expand_variables_in_list_with_subdict():

    to_expand = ["abc", "expanded_{foo}", "{foo2[a]}_expanded"]

    obtained = expand_variables(to_expand, {"foo": "bar", "foo2": {"a": "subdir"}})

    assert obtained[0] == "abc"
    assert obtained[1] == "expanded_bar"
    assert obtained[2] == "subdir_expanded"


def test_expand_variables_in_dict():

    to_expand = {"a": "abc", "b": "expanded_{foo}", "c": "{foo2}_expanded"}

    obtained = expand_variables(to_expand, {"foo": "bar", "foo2": "bar2"})

    assert obtained["a"] == "abc"
    assert obtained["b"] == "expanded_bar"
    assert obtained["c"] == "bar2_expanded"


def test_build_final_command_file_only():

    cmd, _ = build_final_command("cmd", {}, {}, {}, files=["file"])
    assert cmd == "pydcop cmd file"


def test_build_final_command_with_global_options():

    cmd, _ = build_final_command(
        "cmd", {}, {"logs": "log_file.conf", "timeout": "10"}, {}, files=["file"]
    )
    assert cmd == "pydcop --logs log_file.conf --timeout 10 cmd file"


def test_build_final_command_with_command_options():

    cmd, _ = build_final_command(
        "solve",
        {},
        {},
        {"algo": "dsa", "algo_params": {"variant": "A"}},
        files=["file"],
    )
    assert cmd == "pydcop solve --algo dsa --algo_params variant:A file"


def test_build_final_command_with_command_options_and_expansion():

    cmd, _ = build_final_command(
        "solve",
        {},
        {},
        {"algo": "dsa", "algo_params": {"variant": "A"}},
        files=["file_{algo_params[variant]}.yaml"],
    )
    assert cmd == "pydcop solve --algo dsa --algo_params variant:A file_A.yaml"


@patch("pydcop.commands.batch.run_batch")
def test_run_batches_iteration_only(mock_run_batch):

    definition = """
sets:
  set1:
     iterations: 5
batches:
  batch1:
    command: test
    """
    conf = yaml.load(definition, Loader=yaml.FullLoader)
    run_batches(conf, simulate=False)

    assert mock_run_batch.call_count == 5


@patch("pydcop.commands.batch.run_batch")
def test_run_batches_direct_and_iteration(mock_run_batch):
    with tempfile.TemporaryDirectory() as tmpdirname:

        definition = f"""
sets:
  set1:
     path: {tmpdirname}
     iterations: 3
batches:
  batch1:
    command: test
    """
        conf = yaml.load(definition, Loader=yaml.FullLoader)
        run_batches(conf, simulate=False)

        assert mock_run_batch.call_count == 3


@patch("pydcop.commands.batch.run_cli_command")
def test_generate_ising_5_iteration(run_mock):
    definition = """
sets:
 set1: 
   iterations: 5

batches: 
  batch1:
    command: generate ising
    command_options: 
      row_count: 3
      col_count: 3
    global_options:
      output: ising_{iteration}.yaml  
    """
    batches_def = yaml.load(definition, Loader=yaml.FullLoader)
    batch_module.run_batches(batches_def, simulate=False)

    assert run_mock.call_count == 5
    print(run_mock.calls)
    run_mock.assert_has_calls(
        [
            call(
                "pydcop --output ising_0.yaml generate ising --col_count 3 --row_count 3",
                "",
                None,
            ),
            call(
                "pydcop --output ising_1.yaml generate ising --col_count 3 --row_count 3",
                "",
                None,
            ),
            call(
                "pydcop --output ising_2.yaml generate ising --col_count 3 --row_count 3",
                "",
                None,
            ),
            call(
                "pydcop --output ising_3.yaml generate ising --col_count 3 --row_count 3",
                "",
                None,
            ),
            call(
                "pydcop --output ising_4.yaml generate ising --col_count 3 --row_count 3",
                "",
                None,
            ),
        ],
        any_order=True,
    )


@patch("pydcop.commands.batch.run_cli_command")
def test_generate_ising_variable_row(run_mock):
    definition = """
sets:
 set1: 
   iterations: 1

batches: 
  batch1:
    command: generate ising
    command_options: 
      row_count: [3, 4]
      col_count: 3
    global_options:
      output: ising_{iteration}_{row_count}_{col_count}.yaml  
    """

    batches_def = yaml.load(definition, Loader=yaml.FullLoader)
    batch_module.run_batches(batches_def, simulate=False)

    assert run_mock.call_count == 2

    run_mock.assert_has_calls(
        [
            call(
                "pydcop --output ising_0_3_3.yaml generate ising --col_count 3 --row_count 3",
                "",
                None,
            ),
            call(
                "pydcop --output ising_0_4_3.yaml generate ising --col_count 3 --row_count 4",
                "",
                None,
            ),
        ],
        any_order=True,
    )


@patch("pydcop.commands.batch.run_cli_command")
def test_generate_ising_variable_row_with_dir(run_mock):
    definition = """
sets:
 set1: 
   iterations: 1

batches: 
  batch1:
    command: generate ising
    current_dir: ~/tmp/ising/row{row_count}_col{col_count}/
    command_options: 
      row_count: [3, 4]
      col_count: 3
    global_options:
      output: ising.yaml  
    """

    batches_def = yaml.load(definition, Loader=yaml.FullLoader)
    batch_module.run_batches(batches_def, simulate=False)

    assert run_mock.call_count == 2
    run_mock.assert_has_calls(
        [
            call(
                "pydcop --output ising.yaml generate ising --col_count 3 --row_count 3",
                "~/tmp/ising/row3_col3/",
                None,
            ),
            call(
                "pydcop --output ising.yaml generate ising --col_count 3 --row_count 4",
                "~/tmp/ising/row4_col3/",
                None,
            ),
        ],
        any_order=True,
    )


@mock.patch("pydcop.commands.batch.run_cli_command")
def test_solve_variable_row_with_dir(run_mock, tmpdir):
    dir_path = str(tmpdir.realpath())
    definition = (
        """
sets:
 set1:
   path: """
        + dir_path
        + """/*.yaml 
   iterations: 1

batches: 
  batch1:
    command: solve
    current_dir: ~/tmp/ising/{algo}/
    command_options: 
      algo: dsa
    """
    )

    tmpdir.join("pb1.yaml").write("")
    tmpdir.join("pb2.yaml").write("")

    batches_def = yaml.load(definition, Loader=yaml.FullLoader)
    batch_module.run_batches(batches_def, simulate=False)

    run_mock.assert_has_calls(
        [
            call(
                "pydcop solve --algo dsa " + dir_path + "/pb1.yaml",
                "~/tmp/ising/dsa/",
                None,
            ),
            call(
                "pydcop solve --algo dsa " + dir_path + "/pb2.yaml",
                "~/tmp/ising/dsa/",
                None,
            ),
        ],
        any_order=True,
    )
