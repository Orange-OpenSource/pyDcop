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

When running a batch, each job that ran without error is registered in a `progress`
file, named after the the batches description file.
At startup, the ``batch`` command look for such a file, and skip and jobs that
has been registered. This allow resuming an interrupted batch.

Once all the batches have run completely (not stopped and without error), the file is renamed
to "done_<batches_description_file>_<date>"  where <date> is the date and time of the end of the batch.
If you really want to re-run an interrupted batch from scratch, you must delete the `progress`
file.

TODO: in simulate, emit warning if some path / file overlap
TODO: run in parallel

"""
import datetime
import glob
import logging
import shutil
import re
import os
import signal
import pathlib

from subprocess import (
    check_output,
    STDOUT,
    CalledProcessError,
    TimeoutExpired,
    CompletedProcess,
    Popen,
    PIPE,
)
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


progress_file = None


def run_cmd(args):

    with open(args.bench_file, mode="r", encoding="utf-8") as f:
        bench_def = yaml.load(f, Loader=yaml.FullLoader)

    # Search for already run jobs in a 'progress' file, if any.
    # Any job listed in this file will not be re-executed.
    global progress_file
    batch_file = os.path.splitext(os.path.basename(args.bench_file))[0]
    progress_file = f"progress_{batch_file}"

    if os.path.exists(progress_file):
        with open(progress_file, encoding="utf-8", mode="r") as f:
            jobs = [line[5:-2] for line in f.readlines() if line.startswith("JID: ")]
        jobs = set(jobs)
    else:
        with open(progress_file, encoding="utf-8", mode="a") as f:
            now = datetime.datetime.now()
            f.write(f"{batch_file}_{now:%Y%m%d_%H%M}\n")
        jobs = set()

    run_batches(bench_def, args.simulate, jobs)

    # As everything went well, we can rename the progress file
    now = datetime.datetime.now()
    shutil.move(progress_file, f"done_{batch_file}_{now:%Y%m%d_%H%m}")


global pbar


def run_batches(batches_definition, simulate: bool, jobs=None):
    jobs = set() if not jobs else jobs
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
            set_estimate * batch_estimate for batch_estimate in batch_estimates
        )
    logger.debug(f"Estimated number of job: {jobs_count}")
    with tqdm.tqdm(total=jobs_count, desc="Progress") as bar:
        global pbar
        pbar = bar
        for set_name in problems_sets:
            pb_set = problems_sets[set_name]
            if "env" in pb_set:
                context.update(pb_set["env"])
            context["set"] = set_name
            logger.debug("Starting set %s", set_name)

            iterations = 1 if "iterations" not in pb_set else pb_set["iterations"]

            if "path_re" in pb_set:
                extras_files = (
                    pb_set["extras_files"] if "extras_files" in pb_set else []
                )
                matched_paths = list_path_re(pb_set["path_re"])
                files, extras, match_contexts, paths = [],[],[], []
                for p, p_context in matched_paths:
                    p_files, p_extras, p_match_contexts = input_files_re(
                        p, pb_set["file_re"], extras_files
                    )
                    [m.update(p_context) for m in p_match_contexts ]
                    files.extend(p_files)
                    extras.extend(p_extras)
                    match_contexts.extend(p_match_contexts)
                    paths.extend([ p for _ in files])

                for file_path, extra_files, match_context, base_path in zip(
                    files, extras, match_contexts, paths
                ):

                    file_context = context.copy()
                    file_context.update(match_context)
                    file_path = os.path.join(base_path, file_path)
                    extra_path = [os.path.join(base_path, e) for e in extra_files]
                    run_batch_for_files(
                        file_path,
                        extra_path,
                        file_context,
                        iterations,
                        batches,
                        global_options,
                        simulate,
                    )


            elif "path" in pb_set and "file_re" not in pb_set:

                for file_path in input_files_glob(pb_set["path"]):
                    run_batch_for_files(
                        file_path,
                        [],
                        context,
                        iterations,
                        batches,
                        global_options,
                        simulate,
                    )
            elif "path" in pb_set and "file_re" in pb_set:
                extras_files = (
                    pb_set["extras_files"] if "extras_files" in pb_set else []
                )
                files, extras, match_contexts = input_files_re(
                    pb_set["path"], pb_set["file_re"], extras_files
                )
                for file_path, extra_files, match_context in zip(
                    files, extras, match_contexts
                ):

                    file_context = context.copy()
                    file_context.update(match_context)
                    file_path = os.path.join(pb_set["path"], file_path)
                    extra_path = [os.path.join(pb_set["path"], e) for e in extra_files]
                    run_batch_for_files(
                        file_path,
                        extra_path,
                        file_context,
                        iterations,
                        batches,
                        global_options,
                        simulate,
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


def input_files_glob(path_glob: str) -> List[str]:
    """
    Find files matching a glob expression.

    Parameters
    ----------
    path_glob: str
        unix style glob expression (e.g. '/home/user/foo/bar*.json')

    Returns
    -------

    """
    path_glob = os.path.abspath(os.path.expanduser(path_glob))
    logger.debug("Looking for files in %s", path_glob)
    return list(glob.iglob(path_glob))


def list_path_re(path_re):
    p = pathlib.Path(path_re)
    matches = _path_matches(list(p.parts), {})
    return matches


def _path_matches(parts, context):
    result = []
    p1 = pathlib.Path(parts[0])

    children = list(p1.iterdir())

    p2 = parts[1]
    for child in children:
        m = re.match(p2, str(child.name))
        if m:
            if len(parts) > 2:
                entry_context = context.copy()
                entry_context.update(m.groupdict())
                result.extend(
                    _path_matches([p1 / m.group()] + parts[2:], entry_context))
            else:
                entry_context = context.copy()
                entry_context.update(m.groupdict())
                result.append((p1 / m.group(), entry_context))
    return result


def input_files_re(
    path: str, file_re: str, extra_paths: List[str]
) -> Tuple[List[str], List[List[str]], List[Dict]]:
    """

    Parameters
    ----------
    path: str

    file_re: str
        regexp to find main input files
    extra_paths: list of str
        list of file name templates

    Returns
    -------
    files:
        a list of input files
    extras:
        a list containing one list of extra files for each input file
    match_contexts:
        dictionary of match groups
    """
    path = os.path.abspath(os.path.expanduser(path))

    file_re = os.path.basename(file_re)

    matches = []
    all_files = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                all_files.append(entry.name)
                m = re.match(file_re, entry.name)
                if m:
                    logger.debug(f"found file matching re {file_re} : {entry.name}")
                    matches.append(m)

    found_files = []
    found_extras = []
    match_contexts = []
    for m in matches:
        groups = m.groupdict()
        main_file = m.group()
        extra_files = []
        for extra in extra_paths:
            extra = extra.format(**groups)
            if extra not in all_files:
                logger.debug(f"Could not find expected extra file {extra}")
                break
            else:
                logger.debug(f"Found expected extra file {extra}")
                extra_files.append(extra)
        else:
            found_files.append(str(main_file))
            found_extras.append(extra_files)
            match_contexts.append(groups)
    return found_files, found_extras, match_contexts


def estimate_set(set_def: Dict) -> int:
    iterations = 1 if "iterations" not in set_def else set_def["iterations"]

    if "path_re" in set_def:
        matched_paths = list_path_re(set_def["path_re"])
        t = 0
        for p, p_context in matched_paths:
            p_files, p_extras, p_match_contexts = input_files_re(
                p, set_def["file_re"], []
            )
            t += len(p_files)
        return t

    elif "path" in set_def and "file_re" not in set_def:
        file_count = len(input_files_glob(set_def["path"]))
        logger.debug(f"Found {file_count} input to handle")
        return file_count * iterations
    elif "path" in set_def and "file_re" in set_def:
        extras_files = set_def["extras_files"] if "extras_files" in set_def else []

        file_count = len(
            input_files_re(set_def["path"], set_def["file_re"], extras_files)[0]
        )
        logger.debug(f"Found {file_count} input to handle")
        return file_count * iterations
    else:
        return iterations


def estimate_batch(batch_def):
    command_options = (
        batch_def["command_options"] if "command_options" in batch_def else {}
    )
    if command_options:
        command_options = regularize_parameters(command_options)
        return len(parameters_configuration(command_options))
    else:
        return 1


def run_batch_for_files(
    file_path, extra, context, iterations, batches, global_options, simulate
):
    context["file_path"] = file_path
    context["dir_path"] = os.path.dirname(file_path)
    context["file_basename"] = os.path.basename(file_path)
    context["file_name"] = os.path.splitext(os.path.basename(file_path))[0]

    logger.debug("handling file %s", file_path)

    files = [file_path] + extra

    for iteration in range(iterations):
        context["iteration"] = str(iteration)
        logger.debug(
            f"Iteration {iteration} for {file_path} of set {context['set']} with {files}"
        )

        for batch in batches:
            run_batch(batches[batch], context, global_options, files, simulate=simulate)


def run_batch(
    batch_definition: Dict,
    context: Dict[str, str],
    global_options: Dict[str, str],
    files: List[str] = None,
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
            files=files,
        )
        pbar.update(1)
        if simulate:
            if command_dir:
                print(f"cd {command_dir}")
            print(cli_command)
        else:
            jid = job_id(context, command_option_combination)
            if jid not in context["jobs"]:
                log_cmd(cli_command, command_dir)
                if "timeout" in global_options:
                    timeout = int(global_options["timeout"]) + 20
                else:
                    timeout = None
                try:
                    run_cli_command(cli_command, command_dir, timeout)
                except TimeoutExpired as te:
                    global progress_file
                    if progress_file:
                        with open(progress_file, encoding="utf-8", mode="a") as f:
                            f.write(f"TIMEOUT: {jid} \n")
                            f.write(f"JID: {jid} \n")
                            now_time = datetime.datetime.time(datetime.datetime.now())
                            f.write(f"END: {now_time} \n\n")

                register_job(jid)
            else:
                logger.warning(f"Skipping already registered job {id}")


def register_job(jid):
    global progress_file
    if progress_file:
        with open(progress_file, encoding="utf-8", mode="a") as f:
            f.write(f"JID: {jid} \n")
            now_time = datetime.datetime.time(datetime.datetime.now())
            f.write(f"END: {now_time} \n\n")


def log_cmd(cmd_str, command_dir):
    global progress_file
    if progress_file:
        with open(progress_file, encoding="utf-8", mode="a") as f:
            now_time = datetime.datetime.time(datetime.datetime.now())
            f.write(f"START: {now_time} \n")
            f.write(f"CD: {command_dir} \n")
            f.write(f"CMD: {cmd_str} \n")


def job_id(context: dict, combination: dict):
    if "file_name" in context:
        return f"{context['set']}_{context['file_name']}_{context['iteration']}_{combination}"
    else:
        return f"{context['set']}__{context['iteration']}_{combination}"


def run_cli_command(cli_command: str, command_dir: str, timeout):
    with cd_and_create(command_dir):
        try:
            check_output_group_kill(
                cli_command,
                stderr=STDOUT,
                shell=True,
                universal_newlines=True,
                timeout=timeout,
            )
        except CalledProcessError as cpe:
            # Dump output for diagnosis
            with open("cmd_error.log", mode="w", encoding="utf-8") as ef:
                ef.write(
                    f"When running:\n"
                    f" * command: {cli_command}\n"
                    f" * in dir: '{command_dir}'\n\n"
                    f"Error:  \n   {cpe} \n\n"
                )
                ef.write(f"Command returned: \n\n{cpe.output}")
            raise


def check_output_group_kill(*popenargs, timeout=None, **kwargs):
    """
    Custom check_output implementation that kill the whole process tree instead of
    simply it's head.

    Idea taken from
    https://stackoverflow.com/questions/36952245/subprocess-timeout-failure
    """
    if "stdout" in kwargs:
        raise ValueError("stdout argument not allowed, it will be overridden.")

    input = "" if kwargs.get("universal_newlines", False) else b""

    kwargs["stdin"] = PIPE

    with Popen(*popenargs, **kwargs, stdout=PIPE, preexec_fn=os.setsid) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except TimeoutExpired:
            process.kill()
            os.killpg(process.pid, signal.SIGKILL)  # send signal to the process group
            stdout, stderr = process.communicate()
            raise TimeoutExpired(process.args, timeout, output=stdout, stderr=stderr)
        except:
            process.kill()
            process.wait()
            raise
        retcode = process.poll()
        if retcode:
            raise CalledProcessError(
                retcode, process.args, output=stdout, stderr=stderr
            )

    return CompletedProcess(process.args, retcode, stdout, stderr)


def build_final_command(
    command: str,
    context: Dict[str, str],
    global_options: Dict[str, str],
    command_option_combination: Dict,
    current_dir: str = "",
    files: List[str] = None,
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
        parts.extend(files_option)

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
        self.target_path = os.path.expanduser(target_path)

    def __enter__(self):
        self.previous_path = os.getcwd()
        if not self.target_path:
            return
        if not os.path.exists(self.target_path):
            os.makedirs(self.target_path)
        os.chdir(self.target_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.previous_path)
