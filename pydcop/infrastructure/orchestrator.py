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


import threading
from queue import Queue
from time import perf_counter
from typing import Dict, Tuple, Callable
from typing import List
from typing import Optional, Any

from collections import defaultdict

import yaml

from pydcop.algorithms import AlgoDef, filter_assignment_dict, ComputationDef
from pydcop.computations_graph.objects import ComputationGraph
from pydcop.dcop.dcop import DCOP
from pydcop.dcop.scenario import Scenario
from pydcop.distribution.objects import Distribution
from pydcop.infrastructure.agents import Agent, AgentException
from pydcop.infrastructure.communication import CommunicationLayer, MSG_MGT
from pydcop.infrastructure.computations import Message, message_type, \
    MessagePassingComputation
from pydcop.infrastructure.discovery import Directory, UnknownAgent
from pydcop.reparation.removal import _removal_candidate_agents, \
    _removal_orphaned_computations, _removal_candidate_agt_info

ORCHESTRATOR = 'orchestrator'
ORCHESTRATOR_MGT = '_mgt_orchestrator'


class Orchestrator(object):
    """
    Centralized organisation of the set of agents used to solve a dcop.

    Notes
    -----
    Central orchestration is only used for bootstrapping the system and to
    collect metrics.

    As the orchestrator will generally run in a separate process, it uses an
    agent object and communicates with other agents using messages.

    Main responsibilities:
     * deploying the computations
     * collecting metrics
     * running and stopping agents (note that the orchestrator does not
     create nor start agents, it just request them to run their computations)

    A typical use scenario:
     * create and start the agents for the dcop (thread or process based)
     * create the orchestrator, giving him the dcop and its distribution on
     agents
     * deploy the computations
     * run the computations
     * stop everything

    Examples
     --------

        orchestrator.start()
        orchestrator.deploy_computations()
        orchestrator.start_replication() # only needed for resilient system
        orchestrator.run()`
        orchestrator.stop_agents()`
        orchestrator.stop()`

    Parameters
    ----------
    algo: AlgoDef,
        algorithm used to solve the dcop
    cg: ComputationGraph,
        computation graph
    agent_mapping: Distribution,
        initial distribution of computations on agents
    comm: CommunicationLayer,
        An instance of communication layer object
    dcop: DCOP
        The DCOP
    infinity=float
        infinity
    collector: Queue
        A queue used to collect metrics
    collect_moment: str
        metrics collection mode (e.g. 'value_change')

    """

    def __init__(self, algo: AlgoDef, cg: ComputationGraph,
                 agent_mapping: Distribution,
                 comm: CommunicationLayer,
                 dcop: DCOP,
                 infinity=float('inf'),
                 collector: Queue=None,
                 collect_moment: str='value_change'):
        self._own_agt = Agent(ORCHESTRATOR, comm)
        self.directory = Directory(self._own_agt.discovery)
        self._own_agt.add_computation(self.directory.directory_computation)
        self._own_agt.discovery.use_directory(ORCHESTRATOR,
                                              self._own_agt.address)
        self.discovery = self._own_agt.discovery
        self.messaging = comm.messaging

        self.logger = self._own_agt.logger
        self.dcop = dcop

        # For scenario execution
        self._events_iterator = None
        self._event_timer = None  # type: threading.Timer
        self._timeout_timer = None

        self._stopping = threading.Event()

        self.mgt = AgentsMgt(algo, cg, agent_mapping, dcop,
                             self._own_agt, self, infinity, collector=collector,
                             collect_moment=collect_moment)

    @property
    def address(self):
        return self._own_agt.address

    def set_error_handler(self, callback: Callable):
        """
        Set a callback that will be called if the orchestrator thread stops
        due to an unexpected error.

        Parameters
        ----------
        callback: a signle-argument callable
            the callback must accept a single argument, which will the the
            exception that caused the orchestrator thread to stop.
        """
        self._own_agt.on_fatal_error = callback

    def start(self):
        """
        Start the orchestrator.

        Notes
        -----
        The orchestrator, and it's directory, must be
        started in order to receive registration messages, which means you
        must always start the orchestrator before the agents.
        """
        self._own_agt.start()
        self._own_agt.run(self.directory.directory_computation.name)

        self._own_agt.add_computation(self.mgt, ORCHESTRATOR_MGT)
        self._own_agt.run(self.mgt.name)

    def stop(self):
        """
        Stop the orchestrator.

        Notes
        -----
        Once stopped, the orchestrator will not receive
        nor send any new message. This means that agents must be stopped before
        stopping the orchestrator.

        """
        self.logger.info('Requesting orchestrator to stop')
        self._own_agt.stop()
        if self._event_timer is not None:
            self._event_timer.cancel()
            self._event_timer = None

    def deploy_computations(self, once_registered=True):
        """
        Deploy the computation for the dcop.

        The computations are deployed according to the Computation Graph
        and the initial distribution (given to the Orchestrator's constructor).

        Parameters
        ----------
        once_registered: bool
            wait until all agents have registered before starting deployment.
        """

        if once_registered:
            self.logger.info('Waiting for all registration before deploying '
                             'computations')
            self.mgt.all_registered.wait()
        self.logger.info('deploying computations')
        self._mgt_method('_orchestrator_deploy_computations', None)

    def start_replication(self, k_target: int):
        """
        Ask all agents to replicate their computations.

        Notes
        -----
        deploy_computations must be called before, otherwise agents have no
        computation to replicate !

        Parameters
        ----------
        k_target: int
            number of replica for each computation (aka resiliency level).
        """
        # We must be sure computations have been deployed first
        self.logger.info('Waiting until agents are ready to run before '
                         'starting replication')
        self.mgt.ready_to_run.wait()
        self.mgt.ready_to_run = threading.Event()
        self.logger.info('Starting replication')
        self._mgt_method('_orchestrator_start_replication', k_target)

    def run(self, scenario: Scenario=None,
            timeout: Optional[float]=None):
        """Run the DCOP, with a scenario if given.

        When `run()` is called, the orchestrator asks all orchestrated agents to
        start their computations. If the agents are not ready, the orchestrator
        automatically waits until agents are ready (i.e. computations have
        been deployed).
        This methods returns immediately, if you need to wait until the run
        is finished, especially when using a timeout, use wait_ready.

        Parameters
        ----------
        scenario: Scenario
            an optional Scenario object whose events will be injected into
            the system.
        timeout: float
            time, in seconds, after which all agents, and the orchestrator
            itself, must be stopped.

        """
        self.logger.info('Waiting until agents are ready to run')
        self.mgt.ready_to_run.wait()
        self.logger.info('Requesting agents to run')
        self._mgt_method('_orchestrator_run_computations', None)

        if timeout is not None:
            self._timeout_timer = threading.Timer(timeout,
                                                  self._on_timeout)
            self._timeout_timer.start()
            self.mgt.ready_to_run = threading.Event()

        if scenario is not None:
            self._events_iterator = iter(scenario)
            self._process_event()

    def stop_agents(self, timeout: float):
        self.logger.info('Requesting all agents to stop')
        self._stopping.set()
        # WARNING: must NOT access the mgt directly, all its action
        # must be done in the agent's tread. That's the reason we use a msg
        # here. It must have MSG_MGT type to have higher priority, in case the
        # orchestrator's queue is full of other messages.
        if self._event_timer is not None:
            self._event_timer.cancel()
            self._event_timer = None
        self._mgt_method('_orchestrator_stop_agents', None)
        self.mgt.wait_stop_agents(timeout)

        self.mgt.ready_to_run.set()

    def current_global_cost(self):
        return self.mgt.current_global_cost()

    def current_solution(self):
        return self.mgt.current_solution()

    def end_metrics(self):
        return self.mgt.global_metrics('END', perf_counter())

    def wait_ready(self):
        """Blocks until the Orchestrator is ready to perform another action.

        This can be used to wait until the dcop has finished running,
        for example when using a timeout.

        Notes
        -----
        When calling `wait_ready` after `run()` with a tiemou, you may be
        blocked for a longer time than the timeout, as orchestrator also wait
        until all agents have stopped.

        Examples
        --------
        orchestrator.run(timeout=5)
        orchestrator.wait_ready()


        """
        self.mgt.ready_to_run.wait()
        return self._own_agt.is_running and not self._stopping.is_set()

    def _process_event(self):
        try:
            evt = next(self._events_iterator)
        except StopIteration:
            self.logger.info("All events processed for scenario")
            self._events_iterator = None
            return

        if evt.is_delay:
            self.logger.info('Delay: wait %s s for next event', evt.delay)
            self._event_timer = threading.Timer(evt.delay, self._process_event)
            self._event_timer.start()

        else:
            self.logger.info('posting event to mgt %s', evt)
            self._mgt_method('_orchestrator_scenario_event', evt)
            self._process_event()

    def _mgt_method(self, method: str, arg: Any):
        self.messaging.post_msg(
            ORCHESTRATOR_MGT, ORCHESTRATOR_MGT,
            Message(method, arg), msg_type=MSG_MGT)

    def _on_timeout(self):
        self.stop_agents(5)
        self.stop()
        self.mgt.ready_to_run.set()


################################################################################
#  Orchestration messages definition

DeployMessage = message_type('deploy', ['comp_def'])

RunAgentMessage = message_type('run_computations', ['computations'])

ReplicateComputationsMessage = message_type('replication', ['k'])

ComputationReplicatedMessage = message_type('replicated',
                                            ['agent', 'replica_hosts'])

PauseMessage = message_type('pause_computations', ['computations'])

ResumeMessage = message_type('resume_computations', ['computations'])


# StopAgentMessage is sent by the orchestrator to orchestrated agents to
# indicate that they must stop.
StopAgentMessage = message_type('stop', [])

# AgentStoppedMessage is sent by an orchestrated agent to the orchestrator
# to indicate that it has stopped as requested by a StopAgentMessage.
# This messages includes all metrics gathered during the run of the agent.
AgentStoppedMessage = message_type('stopped', ['agent', 'metrics'])

# ValueChangeMessage is sent by the orchestrated agent to the the orchestrator
# when one of hosted the computations change its value.
# We need to include the name of the computation in the message because it
# will be sent by a management computation and not by the computation
# actually changing it's selected value
# agent: str, computation: str, value: int, cost: int, cycle: int, metrics
ValueChangeMessage = message_type(
    'value_change',
    ['agent', 'computation', 'value', 'cost', 'cycle', 'metrics'])

CycleChangeMessage = message_type(
    'cycle_change',
    ['agent', 'computation', 'cycle', 'metrics'])

# MetricsMessage is used to send various metric from the agent to the
# orchestrator. It is only sent when using periodic metric collection.
MetricsMessage = message_type('metrics', ['agent', 'metrics'])

# A ComputationFinishedMessage is sent by an agent to inform the orchestrator
# that a computation is finished.
ComputationFinishedMessage = message_type(
    'end_of_computation', ['agent', 'computation'])

# A AgentRemovedMessage is sent to an agent to inform it that it has been
# removed from the system and must stop all operations.
AgentRemovedMessage = message_type('agent_removed', [])

RepairDoneMessage = message_type('repair_done',
                                 ['agent', 'selected_computations'])


class RepairRunMessage(Message):
    """
    Sent by the orchestrator to resilient orchestrated agents to start the
    computations for the repair dcop are ready.

    """
    def __init__(self):
        super().__init__('repair_run', None)

    def __str__(self):
        return 'RepairRunMessage()'

    def __repr__(self):
        return 'RepairRunMessage()'

    def __eq__(self, other):
        return isinstance(other, RepairRunMessage)


class SetupRepairMessage(Message):
    """
    Sent to an agent when the distribution must be repaired.

    This messages contains the list of candidate computation that could be
    hosted on that agent, with all necessary information to instantiate
    computations for a reparation dcop.
    """

    def __init__(self,
                 repair_info: Dict[str, Tuple[List[str],
                                              Dict[str, str],
                                              Dict[str, List[str]]]]):
        super().__init__('setup_repair', None)
        self._repair_info = repair_info

    @property
    def repair_info(self) -> Dict[str, Tuple[List[str],
                                             Dict[str, str],
                                             Dict[str, List[str]]]]:
        return self._repair_info

    def __str__(self):
        return 'SetupRepairMessage({})'.format(self._repair_info)


class RepairReadyMessage(Message):
    """
    Sent by a resilient orchestrated agent to the orchestrator, when the
    computations for the repair dcop are ready.

    Parameters
    ----------

    agent: str
        the name of the agent
    computations: list of str
        a list containing the computations that have been deployed on this
        agent for reparation purposes.

    """

    def __init__(self, agent: str, computations: List[str]):
        super().__init__('repair_ready', None)
        self._agent = agent
        self._computations = computations

    @property
    def agent(self) -> str:
        return self._agent

    @property
    def computations(self) -> List[str]:
        return self._computations

    def __str__(self):
        return 'RepairReadyMessage({}, {})'.format(self._agent,
                                                   self._computations)

    def __repr__(self):
        return 'RepairReadyMessage({}, {})'.format(self._agent,
                                                   self._computations)

    def __eq__(self, other):
        if not isinstance(other, RepairReadyMessage):
            return False
        if self.agent != other.agent:
            return False
        if self.computations != other.computations:
            return False
        return True

################################################################################
#  Orchestration computation


class AgentsMgt(MessagePassingComputation):
    """
    Computation for the orchestrator.

    This computation should only be used with the orchestrator agent. It
    handles agent management messages:
    * waits for register message
    * start
    * deploy
    * stop

    Notes
    -----
    This computation is never used directly, it is automatically created
    and hosted by the Orchestrator. All its methods must be run on the
    orchestrator's agent thread and should thus never be called directly.

    Parameters
    ----------
    algo: AlgoDef
        The algorithm used to solve the DCOP
    cg: ComputationGraph
        The computation graph containing all the computations fro the dcop
    dcop: DCOP
        The dcop
    orchestrator_agent: Agent
        The orchestrator's agent
    infinity: float
        value used to represent infinity for hard constraints
    collector: Queue
        metrics will be posted on this queue.
    collect_moment:
        metrics collection mode
    """

    def __init__(self, algo: AlgoDef, cg: ComputationGraph,
                 agent_mapping: Distribution,
                 dcop: DCOP,
                 orchestrator_agent: Agent, orchestrator: Orchestrator,
                 infinity=float('inf'),
                 collector: Queue=None,
                 collect_moment: str='value_change'):
        super().__init__(ORCHESTRATOR_MGT)
        self._orchestrator_agent = orchestrator_agent
        self._orchestrator = orchestrator
        self.discovery = self._orchestrator_agent.discovery  # type: Discovery

        self._algo = algo
        self.graph = cg
        self._dcop = dcop
        self.infinity = infinity
        self.initial_dist = agent_mapping

        self.logger = orchestrator_agent.logger
        self._msg_handlers = {
            # Messages from OrchestratedAgents manager by this Orchestrator:
            'value_change': self._on_value_change_msg,
            'cycle_change': self._on_cycle_change_msg,
            'metrics': self._on_metrics_msg,
            'end_of_computation': self._on_computation_end_msg,
            'stopped': self._on_agent_stopped_msg,
            'replicated': self._on_computation_replicated_msg,
            'repair_ready': self._on_repair_ready,
            'repair_done': self._on_repair_done
        }

        self._collect_moment = collect_moment
        self._collector = collector

        self._nb_computations = 0
        self.start_time = None

        # Used to store stae of agent: replication | repair_setup |
        # repair_ready | repair_done
        self._agts_state = {}  # type: Dict[str, str]
        self._comps_state = {}  # type: Dict[str, str]

        self.all_registered = threading.Event()
        self.ready_to_run = threading.Event()

        # To wait for agent when stopping
        self._all_agt_stopped = threading.Event()

        # metrics
        # Storing metrics for agent across several cycles :
        # Dict cycle->agent->metrics or value
        self._current_cycle = 1
        self._agt_cycle_metrics = defaultdict(lambda: {})
        self._agent_cycle_values = defaultdict(lambda: {})
        # used to detect the end of a cycle
        self._computation_cycle = defaultdict(lambda: set())

        self.dist_count = 0

    @property
    def type(self):
        return 'mgt'

    def on_start(self):
        """Called when running the computation"""
        for agt in self._dcop.agents:
            self.discovery.subscribe_agent(agt, self._cb_agent_registration)

    def on_message(self, sender_name: str, msg, t):
        try:
            try:
                self._msg_handlers[msg.type](sender_name, msg, t)
            except KeyError:
                # If we don't have an explicit handler for this message, try to
                # interpret it as an indirect method call from the orchestrator.
                # In this case the type of the message is the name of tge method
                # we should call.
                try:
                    handler = getattr(self, msg.type)
                except AttributeError:
                    raise AgentException('No handler for {} on orchestrator '
                                         ''.format(msg.type))
                handler(msg, t)
        except Exception:
            # In case we have an exception while handling message in
            # orchestrator we need to catch it to avoid stopping the
            # orchestartor's thread
            self.logger.error('Critical error in orchestrator while handling '
                              '%s from %s ', msg, sender_name, exc_info=1)
            self._orchestrator.stop_agents(10)
            self.stop()


    def current_global_cost(self):
        # Compute the cost as seen by the agents
        # Note: this is NOT the global cost of the solution, as agents may
        # have an overlapping view on theirs constraints cost. Actually,
        # it is not clear if that value is usefull at all...
        complete = True
        cost = 0
        for c in self.graph.nodes:
            try:
                cost += self._agent_cycle_values[self._current_cycle][
                    c.name][1]
            except KeyError:
                complete = False
            except TypeError:
                pass
        return cost, complete

    def current_solution(self):
        complete = True
        solution = {}
        for c in self.graph.nodes:
            try:
                solution[c.name] = self._agent_cycle_values[
                    self._current_cycle][c.name]
            except KeyError:
                self.logger.info('Could not find value for computation %s',
                                 c.name)
                solution[c.name] = None
                complete = False
        return solution, complete

    def _cb_agent_registration(self, evt: str, agent: str, _):
        # Cb registered to discovery: called for agent events
        if evt == 'agent_added':
            self.logger.info('Receiving registration %s from agent %s', evt,
                             agent)
            missing = []
            for agt in self.initial_dist.agents:
                try:
                    self.discovery.agent_address(agt)
                except UnknownAgent:
                    missing.append(agt)
            if missing:
                self.logger.info('Still waiting for agents registration %s',
                                 missing)
            else:
                self.all_registered.set()
        elif evt == 'agent_removed':
            self.logger.info('Agent %s unregistered from orchestrator', agent)
        else:
            raise Exception('Invalid agent registration notification')

    def _cb_computation_registration(self, evt: str,
                                     computation: str, agent: str):
        # Cb registered to discovery: called for computations events
        self.logger.debug('Receiving computation registration %s: %s on %s',
                          evt, computation, agent)
        if evt == 'computation_added':
            deployed = set(self.discovery.computations())
            expected = set(self.initial_dist.computations)
            if computation not in expected:
                return

            missing = expected - deployed
            if not missing:
                # once all computations have been deployed, ask agents to run
                # them
                self.logger.info('All computations are deployed')
                self.ready_to_run.set()
            else:
                self.logger.info(
                    'Computation %s deployed on %s, still waiting for %s for '
                    'running them ', computation, agent, len(missing))
                self.logger.debug('Missing computation for starting  : %s ',
                                  missing)

    def _cb_replica_registration(self, evt: str, replica: str, agent: str ):
        if evt == 'replica_added':
            self.logger.debug('Replica added for %s on %s, now : %s ',
                              replica, agent,
                              self.discovery.replica_agents(replica))
        elif evt == 'replica_removed':
            self.logger.debug('Replica removed for %s on %s, now : %s ',
                              replica, agent,
                              self.discovery.replica_agents(replica))

    def _on_computation_replicated_msg(self, sender: str,
                                       msg: ComputationReplicatedMessage, _):

        if msg.agent in self._agts_state \
              and self._agts_state[msg.agent] == 'replicating':
            self._agts_state[msg.agent] = 'ready'
            waited = [v for v in self._agts_state
                      if self._agts_state[v] == 'replicating']
            self.logger.info('Agent %s(%s) has finished replicating its '
                             'computations : %s - waiting for %s',
                             msg.agent, sender, msg, waited)
            if not waited:
                self.logger.info('All computations have been replicated')
                self.ready_to_run.set()
        else:
            # We can get these message when a replication was not initiated
            # by the orchestrator, e.g. when re-replicating after a reparation
            self.logger.info('Agent %s has finished replicating its '
                             'migrated computations : %s ', msg.agent, msg)
            pass

    def _on_value_change_msg(self, sender: str, msg: ValueChangeMessage,
                             t: float):
        """
        Called every time a computation hosted by one of the orchestrated
        agents selects a value.

        If the collection mode is 'value_change', the message also contains the
        metrics.
        """
        self.logger.debug('Receiving value change from %s : %s - %s (%s)',
                          msg.computation, msg.value, msg.cost, sender)

        if self._collect_moment == 'cycle_change':
            self._agent_cycle_values[msg.cycle][msg.computation] = \
                (msg.value, msg.cost)

        else:
            self._agent_cycle_values[self._current_cycle][msg.computation] =\
                (msg.value, msg.cost)
            if self._collect_moment == 'value_change':
                if msg.computation not in self._dcop.variables:
                    # only emit metrics for dcop variable (not reparation
                    # variable)
                    return

                self._agt_cycle_metrics[self._current_cycle][msg.agent] =\
                    msg.metrics
                self._emit_metrics(t)

    def _on_cycle_change_msg(self, sender: str, msg: CycleChangeMessage,
                             t: float):
        # Called when receiving a cycle change message from one of the agent.
        # This should only happen when using 'cycle_change' collection mode.
        if self._collect_moment != 'cycle_change':
            self.logger.error('Should not receive CycleChangeMessage from %s  '
                              'when not using the cycle_change collection '
                              'mode', sender)

        else:
            self.logger.debug('Received cycle change from %s : %s ',
                              msg.agent, msg.cycle)

            cycle_end = msg.cycle-1

            if msg.computation in self._computation_cycle[cycle_end]:
                self.logger.error('Metrics received twice for computation %s '
                                  'and cycle %s: \n %s', msg.computation,
                                  cycle_end, msg.metrics)
            else:
                self._computation_cycle[cycle_end].add(msg.computation)
                self._agt_cycle_metrics[cycle_end][msg.agent] = msg.metrics
                if len(self._computation_cycle[cycle_end]) == \
                        self._nb_computations:
                    self._current_cycle = cycle_end

                    if cycle_end > 0:
                        # During a cycle, not all computation select a new
                        # value, we need to get the unchanged values frm the
                        #  previous cycle.
                        vals = self._agent_cycle_values[cycle_end-1].copy()
                        vals.update(self._agent_cycle_values[cycle_end])
                        self._agent_cycle_values[cycle_end] = vals
                        self.logger.debug('Cycle %s is finished : %s',
                                          cycle_end, vals)

                    self._emit_metrics(t)
                    del self._agt_cycle_metrics[cycle_end]
                    del self._computation_cycle[cycle_end]
                else:
                    self.logger.debug('Store metrics for cycle %s on '
                                      'computation %s ',
                                      cycle_end, msg.computation)

    def _on_metrics_msg(self, sender: str, msg: MetricsMessage, t):
        # Called when receiving a metric message from one of the
        # orchestrated agent. The metric message contains the metrics for
        # all the computations hosted by this agent.
        self.logger.info('Received metrics from %s : %s - %s', msg.agent,
                         dict(msg.metrics), sender)
        self._agt_cycle_metrics[self._current_cycle][msg.agent] = msg.metrics

        self._emit_metrics(t)

    def _on_agent_stopped_msg(self, sender: str, msg: AgentStoppedMessage,
                              _: float):
        self.logger.debug('Received stopped from %s : %s - %s', msg.agent,
                          dict(msg.metrics), sender)
        try:
            self._agt_cycle_metrics[self._current_cycle][msg.agent] \
                = msg.metrics
        except ValueError:
            self.logger.warning('Stopped message for an unexpected agent: %s ',
                                msg.agent)

        remaining_agents = self.discovery.agents()
        if remaining_agents:
            self.logger.info('receiving stopped from %s, still waiting for '
                             'agents %s to stop', msg.agent, remaining_agents)
        else:
            self.logger.info('All agents have stopped')
            self._all_agt_stopped.set()

    def _on_computation_end_msg(self, sender: str,
                                msg: ComputationFinishedMessage, _: float):
        """
        Called when an agent informs the orchestrator that a computation is
        finished.
        """
        self.logger.info('Received computation_end from %s : %s - %s',
                         msg.agent, msg.computation, sender)

    def _orchestrator_deploy_computations(self, *_):
        """
        Deploy the computations on the orchestrated agents.

        Agents must have been started before.
        """
        self.logger.info('Deploying computations on all registered agents %s',
                         self.discovery.agents())
        for agt in self.discovery.agents():
            if agt == 'orchestrator':
                continue
            self._deploy_computation(agt)

        # Now we wait for confirmation of all deployments before running the
        # agents. These confirmations will arrive through discovery, on
        # `_cb_computation_registration`

    def _orchestrator_run_computations(self, *_):
        """
        Request all orchestrated agents to start their computations.
        """
        self.logger.info('Request agents to run')
        self.start_time = perf_counter()
        for agt in self.discovery.agents():
            if agt == 'orchestrator':
                continue
            computations = self.initial_dist.computations_hosted(agt)
            self._send_mgt_msg(agt, RunAgentMessage(computations))
            self._agts_state[agt] = 'running'

    def _orchestrator_start_replication(self, msg, *_):
        """
        Requests orchestrated agents to replicate their computation
        """
        self.logger.info('Request agents to start replication from')
        for agt in self.discovery.agents():
            self._send_mgt_msg(agt, ReplicateComputationsMessage(msg.content))
            self._agts_state[agt] = 'replicating'

    def _orchestrator_scenario_event(self, msg: Message, _: float):
        """
        Handler for internal message `scenario_event` from the orchestrator.
        """
        self.logger.debug('Scenario event from : %s',  msg)
        # Pause the current dcop before injecting the event
        for agt in self.discovery.agents():
            self._send_mgt_msg(
                agt, PauseMessage(
                    self.discovery.agent_computations(agt)))

        evt = msg.content  # type: ScenarioEvent
        leaving_agents = []
        for a in evt.actions:
            if a.type == 'add_agent':
                self.logger.info('Event action: Adding agent %s', a)
                agt = a.args['agent']
                # self._msg_sender.post_msg(self.name, agt + '_mgt',
                #                           RunAgentMessage(), MSG_MGT)

            elif a.type == 'remove_agent':
                self.logger.info('Event action: Removing agent %s', a)
                agt = a.args['agent']
                self._send_mgt_msg(agt, AgentRemovedMessage())
                leaving_agents.append(agt)
            else:
                self.logger.error('Unknown event action %s ', a)
                raise ValueError('Unknown event action ' + str(a))

        self._agents_removal(leaving_agents)
        #self._agents_arrival()

    def _agents_removal(self, leaving_agents: List[str]):
        # Now inform other agents of the list of agents that left the system
        # This replace a proper discovery mechanism
        candidates_agents = _removal_candidate_agents(
            leaving_agents, self.discovery)
        orphaned = _removal_orphaned_computations(leaving_agents,
                                                  self.discovery)
        orphaned_replicas = {o: self.discovery.replica_agents(o) for o in
                             orphaned}
        self.logger.info('On removal of agents %s, orphaned computations: %s '
                         'with candidates %s',
                         leaving_agents, orphaned_replicas, candidates_agents)
        for o, hs in orphaned_replicas.items():
            if not hs:
                self.logger.error('Orphaned computation %s has no known '
                                  'replica: will not be repaired', o)
        self._comps_state.update({c : None for c in orphaned})

        # For removal, agents that must be informed are agents that possess a
        # replica of one of the orphaned computation.
        for candidate in candidates_agents:
            info = _removal_candidate_agt_info(
                candidate, leaving_agents, self.graph,
                self.discovery)
            self.logger.debug('Info for candidate agent %s : %s', candidate,
                              info)
            msg = SetupRepairMessage(info)
            self._send_mgt_msg(candidate, msg)
            self._agts_state[candidate] = 'repair_setup'

    def _agents_arrival(self, arrived_agents: List[str]):
        # TODO
            # For arrival,
        #  * agents that are 'near' the newly arrived agent(s)
        #    this definition of 'near' depends on ??? FIXME
        pass

    def _on_repair_ready(self, sender_name: str, msg: RepairReadyMessage, _):
        # Call when receiving a repair_ready msg from an orchestrated agent
        try:
            current_agt_state = self._agts_state[msg.agent]
        except KeyError:
            self.logger.error('Unexpected repair ready message from agent %s '
                              'with no registered state %s : %s',
                              sender_name, msg.agent, msg)
            return

        if current_agt_state== 'repair_setup':
            self._agts_state[msg.agent] = 'repair_ready'
            waited = [v for v in self._agts_state
                      if self._agts_state[v] == 'repair_setup']
            if waited:
                self.logger.info(
                    'Agent %s ready for repair with computations %s, '
                    'waiting for %s', msg.agent, msg.computations, waited)
            else:
                ready = [v for v in self._agts_state
                         if self._agts_state[v] == 'repair_ready']
                self.logger.info(
                    'Agent %s ready for repair with computations %s, '
                    'all agents ready, sending repair_run to %s', msg.agent,
                    msg.computations, ready)
                for agt in ready:
                    self._send_mgt_msg(agt, RepairRunMessage())
                    self._agts_state[agt] = 'repair_run'
        else:
            self.logger.error('Unexpected repair ready message from agent %s '
                              'with state "%s" : %s ', msg.agent,
                              current_agt_state, msg)

    def _on_repair_done(self, sender_name: str, msg: RepairDoneMessage, _):
        self.logger.debug('Repair done on agent %s, selected computations %s',
                         msg.agent, msg.selected_computations)
        try:
            current_agt_state = self._agts_state[msg.agent]
        except KeyError:
            self.logger.error('Unexpected repair done message from agent %s '
                              'with no registered state %s : %s',
                              sender_name, msg.agent, msg)
            return
        if current_agt_state == 'repair_run':
            self._agts_state[msg.agent] = 'repair_done'
            waited = [a for a in self._agts_state
                      if self._agts_state[a] == 'repair_run']
            self._comps_state.update(
                {c: msg.agent for c in msg.selected_computations})

            if waited:
                self.logger.info('Repair done on agent %s, waiting for %s',
                                  msg.agent, waited)
            else:
                # Now that the reparation process is finished, resume,
                # all computation from the original dcop
                ready = [v for v in self._agts_state
                         if self._agts_state[v] == 'repair_done']
                self.logger.info('Repair done on agent %s, all agents done, '
                                  'resuming computations',
                                  msg.agent)

                # Dump current distribution
                dist = { a: self.discovery.agent_computations(a)
                         for a in self.discovery.agents()}
                result = {
                    'inputs': {
                        'dist_algo': 'repair',
                    },
                    'distribution': dist,
                }
                f_name = 'evtdist_{}.yaml'.format(self.dist_count)
                with open(f_name, mode='w', encoding='utf-8') as f:
                    f.write(yaml.dump(result))
                self.dist_count += 1

                # Resume all computation now that everything is ok
                for agent in self.discovery.agents():
                    self._send_mgt_msg(
                        agent,
                        ResumeMessage(self.discovery.agent_computations(agent)))
                    self._agts_state[agent] = 'running'

                # Check if all orphaned computations have been re-hosted.
                lost_orphaned = [c for c, s in self._comps_state.items()
                                 if s is None]
                if lost_orphaned:
                    self.logger.error('Repair process is finished but they '
                                      'are still some orphaned computations !'
                                      ' %s', lost_orphaned)

    def _orchestrator_stop_agents(self, *_):
        """
        Request all orchestrated agents to stop.

        Careful : This must be called from the orchestrator's agent thread.
        """
        active_agents = self.discovery.agents()
        self.logger.info('Request agents to stop %s', active_agents)
        for agt in active_agents:
            if agt == 'orchestrator':
                continue
            self._send_mgt_msg(agt, StopAgentMessage())

    def _deploy_computation(self, agent_id: str):
        """Deploy computations hosted on agent `agent_id` """
        for c in self.initial_dist.computations_hosted(agent_id):
            self._nb_computations += 1
            self.logger.info('Deploying computation %s on %s', c, agent_id)
            comp_def = ComputationDef(self.graph.computation(c),
                                      self._algo)
            self._send_mgt_msg(agent_id, DeployMessage(comp_def))
            self.discovery.subscribe_computation(
                comp_def.node.name, self._cb_computation_registration)
            self.discovery.subscribe_replica(
                comp_def.node.name, self._cb_replica_registration)

    def wait_stop_agents(self, timeout):
        # wait until all agents have indicated they have stopped
        self.logger.info('Orchestrator is waiting for agents to stop')
        self._all_agt_stopped.wait(timeout)

    def global_metrics(self, current_status, t):

        # Current global cost
        agent_values = self._agent_cycle_values[self._current_cycle]
        assignment = {k: agent_values[k][0] for k in agent_values
                      if agent_values[k]}
        # only keep dcop variable to compute the solution cost
        # it might other contain variables used for reparation
        dcop_assignment = filter_assignment_dict(
            assignment, self._dcop.variables.values())
        try:
            violation, cost = self._dcop.solution_cost(dcop_assignment,
                                                       self.infinity)
        except ValueError:
            var_names = set(self._dcop.variables)
            ass_names = set(assignment)
            self.logger.debug(
                'Cannot compute cost for cycle %s, incomplete assignment: '
                'missing var %s in  %s', self._current_cycle,
                (var_names - ass_names), assignment)
            cost, violation = None, None

        # msg stats and activity ratio
        msg_count, msg_size = 0, 0
        activity = 0
        for agt in self._agt_cycle_metrics[self._current_cycle]:
            agt_metrics = self._agt_cycle_metrics[self._current_cycle][agt]
            try:
                for v in agt_metrics['count_ext_msg']:
                    msg_count += agt_metrics['count_ext_msg'][v]
                for v in agt_metrics['size_ext_msg']:
                    msg_size += agt_metrics['size_ext_msg'][v]
                activity += agt_metrics['active'] / \
                    (agt_metrics['active'] + agt_metrics['idle'])
            except KeyError:
                self.logger.warning(
                    'Incomplete metrics for computation %s : %s ',
                    agt, agt_metrics)

        if self._agt_cycle_metrics[self._current_cycle]:
            active_ratio = activity / len(self._agt_cycle_metrics[
                                          self._current_cycle])
        else:
            active_ratio = None

        total_time = t - self.start_time if self.start_time is not None else 0

        global_metrics = {
            'status': current_status,
            'assignment': assignment,
            'cost':  cost,
            'violation':  violation,
            'time': total_time,
            'msg_count': msg_count,
            'msg_size': msg_size,
            'active_ratio': active_ratio,
            'cycle': self._current_cycle,
            'agt_metrics': self._agt_cycle_metrics[self._current_cycle]
        }

        return global_metrics

    def _emit_metrics(self, t):
        if self._collector is not None:
            self._collector.put((t, self.global_metrics('RUNNING', t)))

    def _send_mgt_msg(self, agt, msg):
        self.post_msg('_mgt_' + agt, msg, MSG_MGT)
