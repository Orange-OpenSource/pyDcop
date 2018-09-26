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


"""
.. _pydcop_commands_batch:

pydcop batch
============

Utility command to run batches of cli command, useful for running benchmarks.

Synopsis
--------

::

  pydcop batch <batches_description_file>


Description
-----------

The ``batch`` command run several commands in batch.

It can be used to generate many DCOP of a given kind (using the pydcop generate command)
or to solve set of problems with a predefined set of algorithms and parameters.








"""
import glob
import logging
from subprocess import check_output, STDOUT
from typing import Dict, Tuple, Union, List

import itertools
import yaml

logger = logging.getLogger("pydcop.cli.batch")


def set_parser(subparsers):

    logger.debug("pyDCOP batch ")

    parser = subparsers.add_parser("batch", help="Running benchmarks")
    parser.set_defaults(func=run_cmd)

    parser.add_argument("bench_file", type=str, help="A benchmark definition file")

    parser.add_argument(
        "--simulate",
        default=False,
        action="store_true",
        help="Simulate the bench by printing the commands instead of running them",
    )


def run_cmd(args):

    # TODO : in simulate, emit warning if some path / file overlap
    # TODO: support resuming
    # TODO run in parallel
    # TODO: support creating working dir

    with open(args.bench_file, mode="r", encoding="utf-8") as f:
        bench_def = yaml.load(f)

    run_batches(bench_def, args.simulate)


def run_batches(batches_definition, simulate: bool):
    context: Dict[str, str] = {}
    problems_sets = batches_definition["sets"]
    batches = batches_definition["batches"]
    global_options = (
        batches_definition["global_options"]
        if "global_options" in batches_definition
        else {}
    )
    # initiate global options

    for pb_set_name in problems_sets:
        pb_set = problems_sets[pb_set_name]
        context["set"] = pb_set_name

        iterations = 1 if "iterations" not in pb_set else pb_set["iterations"]
        if "path" in pb_set:

            set_path = pb_set["path"]
            for pb_path in glob.iglob(set_path):

                for iteration in range(iterations):
                    context["iteration"] = str(iteration)

                    for batch in batches:
                        run_batch(
                            batches[batch],
                            context,
                            global_options,
                            pb_path,
                            simulate=simulate,
                        )
        else:
            for iteration in range(iterations):
                context["iteration"] = str(iteration)

                for batch in batches:
                    context["batch"] = batch
                    run_batch(
                        batches[batch], context, global_options, simulate=simulate
                    )


def run_batch(
    batch_definition: Dict,
    context: Dict[str, str],
    global_options: Dict[str, str],
    pb_path: str = None,
    simulate: bool = True,
):
    command = batch_definition["command"]
    global_options.update(batch_definition.get("global_options", {}))

    command_options = batch_definition["command_options"]
    command_options = regularize_parameters(command_options)

    current_dir = (
        batch_definition["current_dir"] if "current_dir" in batch_definition else ""
    )

    for command_option_combination in parameters_configuration(command_options):

        cli_command, command_dir = build_final_command(
            command,
            context,
            global_options,
            command_option_combination,
            current_dir=current_dir,
            files=pb_path,
        )
        if simulate:
            if current_dir:
                print(f"cd {current_dir}")
            print(cli_command)
        else:
            run_cli_command(cli_command, command_dir)


def run_cli_command(cli_command: str, current_dir: str):
    # TODO : add timeout  on top of the command's timeout ?
    output = check_output(cli_command, stderr=STDOUT, shell=True)
    return yaml.load(output.decode(encoding="utf-8"))


def build_final_command(
    command: str,
    context: Dict[str, str],
    global_options: Dict[str, str],
    command_option_combination: Dict,
    current_dir: str = "",
    files: str = None,
) -> Tuple[str, str]:
    context = context.copy()
    context.update(global_options)
    context.update(command_option_combination)

    parts = ["pydcop"]
    global_options = " ".join(
        [build_option_string(k, v) for k, v in global_options.items()]
    )
    global_options = expand_variables(global_options, context)
    if global_options:
        parts.append(global_options)
    parts.append(command)

    command_options = build_option_for_parameters(command_option_combination)
    command_options = expand_variables(command_options, context)
    if command_options:
        parts.append(command_options)

    files_option = expand_variables(files, context)
    if files_option:
        parts.append(files_option)

    full_command = " ".join(parts)

    # pydcop --output ising.yaml  generate ising --row_count 3 --col_count 4 \\
    #        --bin_range 1.6 --un_range 0.05 --intentional --fg_dist

    # pydcop --log log.conf solve --algo mgm --algo_params stop_cycle:20 \\
    #                          --algo_params break_mode:random  \\
    #                          graph_coloring.yaml \\

    return full_command, expand_variables(current_dir, context)


def regularize_parameters(
    yaml_params: Dict
) -> Dict[str, Union[List[str], Dict[str, List[str]]]]:
    """
    Makes sure that parameters values are always represented as a list of string.

    Parameters
    ----------
    yaml_params: dict
        the dict loaded form the yaml parameters definition

    Returns
    -------
    dict:
        A dict where all values are list of string.

    """
    regularized = {}
    for k, v in yaml_params.items():
        if isinstance(v, list):
            regularized[k] = [str(i) for i in v]
        elif isinstance(v, str):
            regularized[k] = [v]
        elif isinstance(v, dict):
            regularized[k] = regularize_parameters(v)
        else:
            regularized[k] = [str(v)]

    return regularized


def parameters_configuration(
    algo_parameters: Dict[str, Union[List[str], Dict]]
) -> List[Dict[str, Union[str, Dict]]]:
    """
    Return a list of dict, each dict representing one parameters combination.

    Parameters
    ----------
    algo_parameters: dict
        a dict of parameters names assiciated with a list of values for this parameter.

    Returns
    -------
     a list of dict, each dict representing one parameters combination

    Examples
    --------

    >>> parameters_configuration({'1': ['a', 'b'], '2': ['c']})
    [{'1': 'a', '2': 'c'}, {'1': 'b', '2': 'c'}]
    """

    param_names, param_values = zip(*algo_parameters.items())

    # expand sub-parameters
    param_values = [
        parameters_configuration(v) if isinstance(v, dict) else v for v in param_values
    ]

    param_combinations = [
        dict(zip(param_names, values_combination))
        for values_combination in itertools.product(*param_values)
    ]

    return param_combinations


def build_option_for_parameters(params: Dict[str, Union[str, Dict]]) -> str:
    options_str = []
    for p, v in params.items():
        if isinstance(v, dict):
            for sub_p, sub_v in v.items():
                options_str.append(build_option_string(p, f"{sub_p}:{sub_v}"))
        else:
            options_str.append(build_option_string(p, v))

    return " ".join(options_str)


def build_option_string(option_name: str, option_value: str = None):
    if option_value is not None:
        return f"--{option_name} {option_value}"
    elif option_value == "":
        return f"--{option_name}"
    return ""


def expand_variables(
    to_expand: Union[str, List, Dict], context: Dict[str, Union[str, Dict]]
):
    if isinstance(to_expand, str):
        return to_expand.format(**context)
    elif isinstance(to_expand, list):
        return [expand_variables(v, context) for v in to_expand]
    elif isinstance(to_expand, dict):
        return {k: expand_variables(v, context) for k, v in to_expand.items()}
    elif not to_expand:
        return ""

    raise ValueError("Invalid input for expand_variables")
