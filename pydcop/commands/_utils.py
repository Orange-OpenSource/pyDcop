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
import csv
import json
import logging
import os
import traceback
from functools import partial
from importlib import import_module

import sys
from queue import Queue, Empty
from types import FunctionType
from typing import List

from pydcop.algorithms import AlgorithmDef, prepare_algo_params, load_algorithm_module

logger = logging.getLogger("pydcop")


def build_algo_def(algo_module, algo_name: str, objective, cli_params: List[str]):
    """
    Build the AlgorithmDef, which contains the full algorithm specification (
    name, objective and parameters)

    Parameters
    ----------
    algo_module: module object
        the module containing the algorithm
    algo_name: str
        name of the algorithm
    objective: str
        'min' or 'max'
    cli_params: dict
        dict containing parameters for the algorithm

    Returns
    -------
    algorithm definition, as an ``AlgorithmDef`` object.
    """

    # Parameters for the algorithm:
    if hasattr(algo_module, "algo_params"):
        params = {}
        if cli_params is not None:
            logger.info("Using cli params for %s : %s", algo_name, cli_params)
            for p in cli_params:
                p, v = p.split(":")
                params[p] = v
        else:
            logger.info("Using default parameters for %s", algo_name)

        try:
            params = prepare_algo_params(params, algo_module.algo_params)
            logger.info("parameters for %s : %s", algo_name, params)

            return AlgorithmDef.build_with_default_param(
                algo=algo_name, params=params, mode=objective
            )

        except Exception as e:
            param_msgs = []
            for param_def in algo_module.algo_params:
                valid_values = (
                    ""
                    if param_def.values is None
                    else "values: {}".format(param_def.values)
                )
                param_msg = "  * {} ({}) default : {}  {} ".format(
                    param_def.name,
                    param_def.type,
                    param_def.default_value,
                    valid_values,
                )
                param_msgs.append(param_msg)
            msg = str(e)
            msg += "\nAvailable parameters for {}:\n".format(algo_name)
            msg += "\n".join(param_msgs)
            _error(msg, e)

    else:
        if cli_params:
            _error("Algo {} does not support any parameter".format(algo_name))
        return AlgorithmDef(algo_name, {}, objective=objective)


# Files for logging metrics
columns = {
    "cycle_change": [
        "cycle",
        "time",
        "cost",
        "violation",
        "msg_count",
        "msg_size",
        "status",
    ],
    "value_change": [
        "time",
        "cycle",
        "cost",
        "violation",
        "msg_count",
        "msg_size",
        "status",
    ],
    "period": ["time", "cycle", "cost", "violation", "msg_count", "msg_size", "status"],
}


def prepare_metrics_files(run, end, mode):
    """
    Prepare files for storing metrics, if requested.
    Returns a cb that can be used to log metrics in the run_metrics file.
    """
    global run_metrics, end_metrics
    if run is not None:
        run_metrics = run
        # delete run_metrics file if it exists, create intermediate
        # directory if needed
        if os.path.exists(run_metrics):
            os.remove(run_metrics)
        else:
            f_dir = os.path.dirname(run_metrics)
            if f_dir and not os.path.exists(f_dir):
                os.makedirs(f_dir)
        # Add column labels in file:
        with open(run_metrics, "w", encoding="utf-8", newline="") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(columns[mode])
        csv_cb = partial(add_csvline, run_metrics, mode)
    else:
        csv_cb = None

    if end is not None:
        end_metrics = end
        if not os.path.exists(os.path.dirname(end_metrics)):
            os.makedirs(os.path.dirname(end_metrics))
        # Add column labels in file:
        if not os.path.exists(end_metrics):
            with open(end_metrics, "w", encoding="utf-8", newline="") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(columns[mode])

    return csv_cb


def add_csvline(file, mode, metrics):
    data = [metrics[c] for c in columns[mode]]
    with open(file, mode="at", encoding="utf-8", newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(data)


def _error(msg, e=None):
    print("Error: {}".format(msg))
    if e is not None:
        print(e)
        tb = traceback.format_exc()
        print(tb)
    sys.exit(2)


def _load_modules(dist, algo):
    dist_module, algo_module, graph_module = None, None, None
    if dist is not None:
        try:
            dist_module = import_module("pydcop.distribution.{}".format(dist))
            # TODO check the imported module has the right methods ?
        except ImportError:
            _error("Could not find distribution method {}".format(dist))

    try:
        algo_module = load_algorithm_module(algo)

        graph_module = import_module(
            "pydcop.computations_graph.{}".format(algo_module.GRAPH_TYPE)
        )

    except ImportError as e:
        _error("Could not find module for algorithm: {}".format(algo), e)
    except Exception as e:
        _error(
            "Error loading algorithm module  and associated "
            "computation graph type for : {}".format(algo),
            e,
        )

    return dist_module, algo_module, graph_module


def collect_tread(collect_queue: Queue, csv_cb):
    while True:
        try:
            t, metrics = collect_queue.get()

            if csv_cb is not None:
                csv_cb(metrics)

        except Empty:
            pass
        # FIXME : end of run ?
