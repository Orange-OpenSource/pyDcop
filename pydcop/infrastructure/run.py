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
import logging
from importlib import import_module
from multiprocessing import Process
from queue import Queue
from typing import Union

from pydcop.algorithms import AlgorithmDef, load_algorithm_module
from pydcop.computations_graph.objects import ComputationGraph
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import AgentDef
from pydcop.distribution.objects import Distribution
from pydcop.infrastructure.communication import InProcessCommunicationLayer, \
    HttpCommunicationLayer
from pydcop.infrastructure.orchestratedagents import OrchestratedAgent
from pydcop.infrastructure.orchestrator import Orchestrator



# FIXME : need better infinity management
INFINITY = 10000


def solve(dcop: DCOP,
          algo_def: Union[str, AlgorithmDef],
          distribution: Union[str, Distribution],
          graph: Union[str, ComputationGraph]=None,
          timeout=5):
    """Solve a dcop in a single process.

    This is simply a convenience method that hides all the complexity of the
    orchestrator, agents creation, etc.

    Parameters
    ----------
    dcop : DCOP
        a DCOP object
    algo_def: string or AlgorithmDef object
        The algorithm that should be used to solve the DCOP. If `algo_def` is
        a string, it is interpreted as an algorithm module in
        the package `pydcop.algorithms`, and this algorithm is used with it's
        default parameters. If `algo_def` is not a string, it must be an
        AlgorithmDef object.
    distribution: either a Distribution object or a string
        If `distribution` is a string, it is interpreted as the name of a module
        in the package pydcop.distribution. This module is the loaded and used
        to generate a distribution of the DCOP on the agents. If `distribution`
        is not a string,  it must be a Distribution object that maps
        computations to agents.
    graph: either a string a ComputationGraph object
        If `graph` is a string, it is interpreted as the name of  module in
        the `pydcop.computations_graph` package ; the module is loaded and
        used to build  computation graph for the dcop. If `graph` is not a
        string, it must be a ComputationGraph object for the given DCOP.
        Ths parameter is optional, if it is not given it is deduced from the
        algorithm.
    timeout:float
        in seconds

    Returns
    -------
    the result

    Examples
    --------

        assignment = solve(dcop, 'maxsum', 'adhoc', timeout=3)

    """

    if isinstance(algo_def, str):
        algo_module = load_algorithm_module(algo_def)
        algo_def = AlgorithmDef.build_with_default_param(
            algo_def, parameters_definitions=algo_module.algo_params,
            mode=dcop.objective
        )
    else:
        algo_module = load_algorithm_module(algo_def.algo)

    if graph is None:
        graph_module = import_module('pydcop.computations_graph.{}'.
                                     format(algo_module.GRAPH_TYPE))
        graph = graph_module.build_computation_graph(dcop)

    elif isinstance(graph, str):
        graph_module = import_module('pydcop.computations_graph.'+graph)
        graph = graph_module.build_computation_graph(dcop)

    if isinstance(distribution, str):
        distrib_module = import_module('pydcop.distribution.' + distribution)
        distribution = distrib_module.distribute(
            graph, dcop.agents.values(),
            computation_memory=algo_module.computation_memory,
            communication_load=algo_module.communication_load)

    orchestrator = run_local_thread_dcop(algo_def, graph, distribution, dcop,
                                         INFINITY)

    try:
        print('Deploy')
        orchestrator.deploy_computations()
        print('start Running')
        orchestrator.run(timeout=timeout)
        print('Running, wait ready')
        orchestrator.wait_ready()
        print('Done')
        return orchestrator.end_metrics()['assignment']

    except Exception:
        orchestrator.stop_agents(5)
        orchestrator.stop()
    finally:
        orchestrator.stop_agents(5)
        orchestrator.stop()


def run_local_thread_dcop(algo: AlgorithmDef,
                          cg: ComputationGraph,
                          distribution: Distribution,
                          dcop: DCOP,
                          infinity,  # FIXME : this has nothing to to here, #41
                          collector: Queue=None,
                          collect_moment: str='value_change',
                          period=None,
                          replication=None,
                          delay=None,
                          uiport=None) -> Orchestrator:
    """Build orchestrator and agents for running a dcop in threads.

    The DCOP will be run in a single process, using one thread for each agent.

    Parameters
    ----------
    algo: AlgorithmDef
        Definition of DCOP algorithm, with associated parameters
    cg: ComputationGraph
        The computation graph used to solve the DCOP with the given algorithm
    distribution: Distribution
        Distribution of the computation on the agents
    dcop: DCOP
        The DCOP instance to solve
    infinity:
        FIXME : remove this!
    collector: queue
        optionnal queue, used to collect metrics
    collect_moment: str
        metric collection configuration : 'cycle_change', 'value_change' or
        'period'
    period: float
        period for collecting metrics, only used we 'period' metric collection
    replication
        replication algorithm,  for resilent DCOP.

    Returns
    -------
    orchestator
        An orchestrator agent that bootstrap dcop agents, monitor them and
        collects metrics.

    See Also
    --------
    Orchestrator, OrchestratedAgent
    run_local_process_dcopb


    """
    agents = dcop.agents
    comm = InProcessCommunicationLayer()
    orchestrator = Orchestrator(algo, cg, distribution, comm, dcop, infinity,
                                collector=collector,
                                collect_moment=collect_moment,
                                collect_period=period,
                                ui_port=uiport)
    orchestrator.start()


    # Create and start all agents.
    # Each agent will register it-self on the orchestrator
    for a_name in dcop.agents:
        if uiport:
            uiport += 1
        comm = InProcessCommunicationLayer()
        agent = OrchestratedAgent(agents[a_name], comm,
                                  orchestrator.address,
                                  metrics_on=collect_moment,
                                  metrics_period=period,
                                  replication=replication,
                                  delay=delay,
                                  ui_port=uiport)
        agent.start()

    # once all agents have started and registered to the orchestrator,
    # computation will be deployed on them and then run.
    return orchestrator


def run_local_process_dcop(algo: AlgorithmDef, cg: ComputationGraph,
                           distribution: Distribution, dcop: DCOP,
                           infinity,  # FIXME : this has nothing to to here, #41
                           collector: Queue=None,
                           collect_moment: str='value_change',
                           period=None,
                           replication=None,
                           delay=None,
                           uiport=None
                           ):

    agents = dcop.agents
    port = 9000
    comm = HttpCommunicationLayer(('127.0.0.1', port))
    orchestrator = Orchestrator(algo, cg, distribution, comm, dcop, infinity,
                                collector=collector,
                                collect_moment=collect_moment,
                                collect_period=period,
                                ui_port=uiport)
    orchestrator.start()

    # Create and start all agents.
    # Each agent will register it-self on the orchestrator
    for a_name in dcop.agents:
        port += 1
        if uiport:
            uiport += 1
        p = Process(target=_build_process_agent, name='p_'+a_name,
                    args=[agents[a_name], port, orchestrator.address],
                    kwargs={'metrics_on': collect_moment,
                            'metrics_period': period,
                            'replication': replication,
                            'delay': delay,
                            'uiport': uiport},
                    daemon=True)
        p.start()

    # once all agents have started and registered to the orchestrator,
    # computation will be deployed on them and then run.
    return orchestrator



def _build_process_agent(agt_def: AgentDef, port, orchestrator_address,
                         metrics_on, metrics_period, replication,
                         delay, uiport):
    comm = HttpCommunicationLayer(('127.0.0.1', port))
    agent = OrchestratedAgent(agt_def, comm, orchestrator_address,
                              metrics_on=metrics_on,
                              metrics_period=metrics_period,
                              replication=replication,
                              delay=delay,
                              ui_port=uiport)

    # Disable all non-error logging for agent's processes, we don't want
    # all agents trying to log in the same console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    root_logger = logging.getLogger('')
    root_logger.setLevel(logging.ERROR)
    root_logger.addHandler(console_handler)

    agent.start()
