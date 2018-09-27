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

When running a batch, each jon that ran without error is registered in a `progress`
file. At startup, the ``batch`` command look for such a file, and skip and jobs that
has been registered. This allow resuming an interrupted batch.

If you really want to re-run a batch from scratch, you must delete the ``progress``
file.






"""
import glob
import logging
from os import chdir, getcwd, makedirs
from os.path import abspath, expanduser, dirname, basename, exists, splitext
from subprocess import check_output, STDOUT
from typing import Dict, Tuple, Union, List

import itertools

import tqdm
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

    # TODO: in simulate, emit warning if some path / file overlap
    # TODO: run in parallel

    with open(args.bench_file, mode="r", encoding="utf-8") as f:
        bench_def = yaml.load(f)

    # Search for already run jobs in a 'progress' file, if any.
    # Any job listed in this file will not be re-executed.
    if exists("progress"):
        with open("progress", encoding="utf-8", mode="r") as f:
            jobs = [job[:-1] for job in f.readlines()]
        jobs = set(jobs)
    else:
        jobs = set()

    run_batches(bench_def, args.simulate, jobs)


global pbar


def run_batches(batches_definition, simulate: bool, jobs):
    context: Dict[str, str] = {"jobs": jobs}
    problems_sets = batches_definition["sets"]
    batches = batches_definition["batches"]
    global_options = (
        batches_definition["global_options"]
        if "global_options" in batches_definition
        else {}
    )

    # Estimates the numbers of jobs:
    # set * nb_file * iteration * combination

    batch_estimates = [estimate_batch(batches[batch]) for batch in batches]
    jobs_count = 0
    for set_name in problems_sets:
        set_estimate = estimate_set(problems_sets[set_name])
        jobs_count += sum(
            set_estimate * batch_estimage for batch_estimage in batch_estimates
        )

    with tqdm.tqdm(total=jobs_count, desc="Progress") as bar:
        global pbar
        pbar = bar
        for set_name in problems_sets:
            pb_set = problems_sets[set_name]
            context["set"] = set_name
            logger.debug("Starting set %s", set_name)

            iterations = 1 if "iterations" not in pb_set else pb_set["iterations"]
            if "path" in pb_set:

                set_path_glob = abspath(expanduser(pb_set["path"]))
                logger.debug("Looking for files in %s", set_path_glob)

                for file_path in glob.iglob(set_path_glob):
                    context["file_path"] = file_path
                    context["dir_path"] = dirname(file_path)
                    context["file_basename"] = basename(file_path)
                    context["file_name"] = splitext(basename(file_path))[0]

                    logger.debug("handling file %s", file_path)

                    for iteration in range(iterations):
                        context["iteration"] = str(iteration)
                        logger.debug(
                            "Iteration %s for file %s of set %s",
                            iteration,
                            basename(file_path),
                            set_name,
                        )

                        for batch in batches:
                            run_batch(
                                batches[batch],
                                context,
                                global_options,
                                file_path,
                                simulate=simulate,
                            )
            else:
                logger.debug(
                    "No files in set %s, running %s iterations ", set_name, iterations
                )

                for iteration in range(iterations):
                    context["iteration"] = str(iteration)

                    logger.debug("Iteration %s of set %s", iteration, set_name)

                    for batch in batches:
                        logger.debug(
                            "Batch %s - iteration %s of set %s",
                            batch,
                            iteration,
                            set_name,
                        )
                        context["batch"] = batch
                        run_batch(
                            batches[batch], context, global_options, simulate=simulate
                        )


def estimate_set(set_def):
    iterations = 1 if "iterations" not in set_def else set_def["iterations"]

    if "path" in set_def:
        set_path_glob = abspath(expanduser(set_def["path"]))
        logger.debug("Looking for files in %s", set_path_glob)

        file_count = len(glob.glob(set_path_glob))
        return file_count * iterations
    else:
        return iterations


def estimate_batch(batch_def):
    command_options = batch_def["command_options"]
    command_options = regularize_parameters(command_options)
    return len(parameters_configuration(command_options))


def run_batch(
    batch_definition: Dict,
    context: Dict[str, str],
    global_options: Dict[str, str],
    file_path: str = None,
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
            files=file_path,
        )
        pbar.update(1)
        if simulate:
            if command_dir:
                print(f"cd {command_dir}")
            print(cli_command)
        else:
            id = job_id(context, command_option_combination)
            if id not in context["jobs"]:
                run_cli_command(cli_command, command_dir)
                register_job(id)
            else:
                logger.warning(f"Skipping already registered job {id}")


def register_job(id):

    with open("progress", encoding="utf-8", mode="a") as f:
        f.write(id + "\n")


def job_id(context: dict, combination: dict):
    if "file_name" in context:
        return f"{context['set']}_{context['file_name']}_{context['iteration']}_{combination}"
    else:
        return f"{context['set']}__{context['iteration']}_{combination}"


def run_cli_command(cli_command: str, command_dir: str):

    with cd_and_create(command_dir):
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
    # We sort by parameter's name so that combinations are always produced in the same
    # order if we run the batch several times.
    param_names, param_values = zip(
        *sorted(algo_parameters.items(), key=lambda x: x[0])
    )
    param_values = [
        sorted(values) if isinstance(values, list) else values
        for values in param_values
    ]

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


class cd_and_create:
    """
    cd_and_create context manager.

    Creates a directory if needed and switch to it.
    When exiting the context mlanager, the initial directory is restored.
    """

    def __init__(self, target_path):
        self.target_path = expanduser(target_path)

    def __enter__(self):
        self.previous_path = getcwd()
        if not self.target_path:
            return
        if not exists(self.target_path):
            makedirs(self.target_path)
        chdir(self.target_path)

    def __exit__(self, etype, value, traceback):
        chdir(self.previous_path)
