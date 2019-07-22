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
from time import perf_counter
from typing import Dict, List, Set

from pydcop.algorithms import ComputationDef
from pydcop.dcop.objects import AgentDef
from pydcop.infrastructure.agents import ResilientAgent
from pydcop.infrastructure.communication import CommunicationLayer, MSG_VALUE, MSG_MGT
from pydcop.infrastructure.computations import (
    MessagePassingComputation,
    Message,
    build_computation,
)
from pydcop.infrastructure.discovery import Address
from pydcop.infrastructure.orchestrator import (
    DeployMessage,
    RunAgentMessage,
    ReplicateComputationsMessage,
    PauseMessage,
    AgentStoppedMessage,
    ValueChangeMessage,
    CycleChangeMessage,
    ComputationFinishedMessage,
    MetricsMessage,
    StopAgentMessage,
    RepairReadyMessage,
    SetupRepairMessage,
    RepairRunMessage,
    ComputationReplicatedMessage,
    RepairDoneMessage,
    ResumeMessage,
    AgentRemovedMessage,
    SetMetricsModeMessage,
)

ORCHESTRATOR = "orchestrator"
ORCHESTRATOR_MGT = "_mgt_orchestrator"


class OrchestratedAgent(ResilientAgent):
    """
    An `OrchestratedAgent` is an agent that can be controlled through
    messages.

    An OrchestratedAgent hosts a specific computation (an
    OrchestrationComputation) that supports management messages for
    deploying, running and pausing computations. These messages will usually
    be sent by an Orchestrator.

    Parameters
    ----------
    agt_def: AgentDef
        definition of the agent that is orchestrated.
    comm: communication layer implementation
    orchestrator_address: Address
        address of the orchestrator
    metrics_on: str
        mode for metrics collection : 'period', 'cycle' 'value_change' or None
    metrics_period: float
        when using metrics_on='period', the periodicity for metrics messages
    delay: int
        An optional delay between message delivery, in second. This delay
        only applies to algorithm's messages and is useful when you want to
        observe (for example with the GUI) the behavior of the algorithm at
        runtime.

    See Also
    --------
    Orchestrator, OrchestrationComputation

    """

    def __init__(
        self,
        agt_def: AgentDef,
        comm: CommunicationLayer,
        orchestrator_address: Address,
        metrics_on: str = None,
        metrics_period: float = None,
        replication: str = None,
        ui_port=None,
        delay: float = None,
    ):
        super().__init__(
            agt_def.name, comm, agt_def, replication, ui_port=ui_port, delay=delay
        )

        # Orchestrator and orchestration computation hosted by it:
        self.discovery.use_directory(ORCHESTRATOR, orchestrator_address)
        self.discovery.register_agent(ORCHESTRATOR, orchestrator_address, publish=False)
        self.discovery.register_computation(
            ORCHESTRATOR_MGT, ORCHESTRATOR, publish=False
        )

        self._mgt_computation = OrchestrationComputation(self)

        self.metrics_on = metrics_on
        if metrics_on == "period":
            self.set_metrics_period(metrics_period)

    def set_metrics_period(self, metrics_period):
        self.set_periodic_action(metrics_period, self._mgt_computation.send_metrics)

    def _on_start(self):
        """
        See Also
        --------
        Agent._on_start

        Returns
        -------
        status

        """
        # Called in the agent's thread when it starts
        if not super()._on_start():
            return False
        self.add_computation(self._mgt_computation)
        self._mgt_computation.start()
        return True

    def _on_computation_value_changed(self, computation: str, value, cost, cycle):
        # Overwritten from Agent
        self._mgt_computation.on_computation_value_changed(
            computation, value, cost, cycle
        )

    def _on_computation_new_cycle(self, computation, *args, **kwargs):
        # Overwritten from Agent
        self._mgt_computation.on_computation_new_cycle(computation, *args, **kwargs)

    def _on_computation_finished(self, comp_name: str, *args, **kwargs):
        # Overwritten from ResilientAgent
        super()._on_computation_finished(comp_name)
        self._mgt_computation.on_computation_finished(comp_name, *args, **kwargs)

    def _on_replication_done(self, replica_hosts: Dict[str, Set[str]]):
        super()._on_replication_done(replica_hosts)
        # Overwritten from ResilientAgent
        self._mgt_computation.on_replication_done(replica_hosts)

    def _on_repair_done(self, selected_computation: List[str], metrics):
        # Overwritten from ResilientAgent
        self._mgt_computation.on_repair_done(selected_computation, metrics)


class OrchestrationComputation(MessagePassingComputation):
    """
    The OrchestrationComputation is used by OrchestratedAgents to answer to
    all messages from the orchestrator.

    Parameters
    ----------
    agent: OrchestratedAgent
        the OrchestratedAgent this OrchestrationComputation is taking care of.
    """

    def __init__(self, agent: OrchestratedAgent):
        super().__init__("_mgt_" + agent.name)
        self.agent = agent
        self.discovery = agent.discovery
        self.logger = logging.getLogger("pydcop.agent.mgt." + agent.name)
        self._handlers = {
            "metrics_mode": self._on_metrics_mode,
            "deploy": self._on_deploy_computations,
            "replication": self._on_replication,
            "run_computations": self._on_run_computations,
            "pause_computations": self._on_pause,
            "resume_computations": self._on_resume,
            "setup_repair": self._on_setup_repair,
            "repair_run": self._on_repair_run,
            "stop": self._on_stop_request,
            # When requested to leave, simply stop
            "agent_removed": self._on_stop_request,
        }

    @property
    def type(self):
        return "mgt"

    def on_start(self):
        # Called when the computation is starting, nothing to do for
        # orchestration computation
        self.logger.debug(
            "Starting orchestration computation fro agent %s", self.agent.name
        )

    def footprint(self):
        return 0

    def on_message(self, sender: str, msg: Message, t: float):
        """
        Called when receiving an management message.
        """
        self._handlers[msg.type](sender, msg, t)

    def _on_metrics_mode(self, sender: str, msg: SetMetricsModeMessage, t: float):
        self.logger.debug("Setting metrics mode from message %s", msg.mode)
        self.agent.metrics_on = msg.mode
        if msg.mode == "period":
            self.agent.set_metrics_period(msg.period)
        self.agent.metrics_on = msg.mode

    def _on_replication(self, sender: str, msg: ReplicateComputationsMessage, t: float):
        self.logger.debug("ReplicateComputationsMessage from %s : %s", sender, msg)
        self.agent.replicate(msg.k)

    def _on_run_computations(self, sender: str, msg: RunAgentMessage, t: float):
        self.logger.debug("RunAgentMessage from %s : %s", sender, msg)
        self.agent.run(msg.computations)

    def _on_stop_request(self, sender: str, msg: StopAgentMessage, t: float):
        self.logger.debug("StopAgentMessage from %s : %s", sender, msg)
        self.send_to_orchestrator(
            AgentStoppedMessage(self.agent.name, self.agent.metrics())
        )
        self.agent.stop()

    def _on_agent_removed(self, sender: str, msg: AgentRemovedMessage, t: float):
        self.logger.debug("AgentRemovedMessage from %s : %s", sender, msg)
        self.send_to_orchestrator(
            AgentStoppedMessage(self.agent.name, self.agent.metrics())
        )
        self.agent.stop()

    def _on_pause(self, sender: str, msg: PauseMessage, t: float):
        self.logger.debug("PauseMessage from %s : %s", sender, msg)
        self.agent.pause_computations(msg.computations)

    def _on_resume(self, sender: str, msg: ResumeMessage, t: float):
        self.logger.debug("ResumeMessage from %s : %s", sender, msg)
        self.agent.unpause_computations(msg.computations)

    def _on_deploy_computations(self, sender: str, msg: DeployMessage, t: float):
        """
        Deploys a new computation on this agent.

        Instantiate the computation, deploy it on the agent and register it
        for replication, if needed.

        Notes
        -----
        We cannot register immediately the neighbor computations as we do
        not know yet which agents are hosting them. Instead we initiate a
        discovery lookup for these computations and they will be registered
        once this lookup is completed.

        Parameters
        ----------

        comp_def: ComputationDef
            Definition of the computation
        """
        comp_def = msg.comp_def
        self.logger.info(
            "Deploying computations %s  on %s", comp_def.node, self.agent.name
        )
        computation = build_computation(comp_def)
        self.agent.add_computation(computation)

    def _on_setup_repair(self, sender: str, msg: SetupRepairMessage, t: float):
        self.logger.info("SetupRepair msg from %s : %s at %s", sender, msg, t)
        repair_computation = self.agent.setup_repair(msg.repair_info)
        computations = [c.name for c in repair_computation.values()]

        self.send_to_orchestrator(RepairReadyMessage(self.agent.name, computations))

    def _on_repair_run(self, sender: str, msg: RepairRunMessage, t: float):
        self.logger.info("RepairRun msg from %s : %s at %s", sender, msg, t)
        self.agent.repair_run()

    def on_computation_value_changed(self, computation: str, value, cost, cycle):
        """
        Called when one of the hosted computation selects a value.

        The value and the corresponding cost is sent to the orchestrator.
        Metrics are also included in the message if the collection mode is
        'value_change'.

        """
        if self.agent.metrics_on == "value_change":
            # Metrics are send with the value message so we do not need to send
            # an extra metrics message, but only when using the 'value_change'
            # mode.
            metrics = self.agent.metrics()
        else:
            metrics = dict()
        value_msg = ValueChangeMessage(
            self.agent.name, computation, value, cost, cycle, metrics
        )
        self.post_msg(ORCHESTRATOR_MGT, value_msg, MSG_MGT)

    def on_computation_new_cycle(self, computation, *args, **kwargs):
        cycle_count, = args
        if self.agent.metrics_on == "cycle_change":
            self.send_to_orchestrator(
                CycleChangeMessage(
                    self.agent.name, computation, cycle_count, self.agent.metrics()
                )
            )

    def on_computation_finished(self, computation):
        self.send_to_orchestrator(
            ComputationFinishedMessage(self.agent.name, computation)
        )

    def on_repair_done(self, selected_computation: List[str], metrics):
        self.send_to_orchestrator(
            RepairDoneMessage(self.agent.name, selected_computation, metrics)
        )

    def on_replication_done(self, replica_hosts: Dict[str, Set[str]]):
        """
        Called when all computations have been replicated.
        Parameters
        ----------

        replica_hosts: a map { computation name -> List of agt name }
            For each active computation hosted by this agent, this map
            contains a list of agents that have been selected to host a
            replica.
        """
        # Message metrics for replication
        # print(f" count {dict(self.agent._messaging.count_ext_msg)}")
        # print(f" size {dict(self.agent._messaging.size_ext_msg)}")
        count_msg = sum(
            v
            for k, v in self.agent._messaging.count_ext_msg.items()
            if k.startswith("_replication")
        )
        size_msg = sum(
            v
            for k, v in self.agent._messaging.size_ext_msg.items()
            if k.startswith("_replication")
        )
        metrics = {
            "count_ext_msg": count_msg,
            "size_ext_msg": size_msg,
        }
        self.send_to_orchestrator(
            ComputationReplicatedMessage(self.agent.name, replica_hosts, metrics)
        )

    def send_metrics(self):
        """
        Send metrics to the orchestrator.
        :return:
        """
        self.send_to_orchestrator(MetricsMessage(self.agent.name, self.agent.metrics()))

    def send_to_orchestrator(self, msg: Message):
        self.post_msg(ORCHESTRATOR_MGT, msg, MSG_MGT)

    def __str__(self):
        return "OrchestrationComputation({}".format(self.name)
