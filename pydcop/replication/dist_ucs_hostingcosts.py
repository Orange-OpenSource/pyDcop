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
Uniform Cost Search distribution

"""
import itertools
import logging
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Tuple, Set, Iterable, Union

from pydcop.algorithms import ComputationDef
from pydcop.infrastructure.agents import Agent
from pydcop.infrastructure.computations import (
    MessagePassingComputation,
    Message,
    register,
)
from pydcop.infrastructure.discovery import Discovery, Address, UnknownComputation
from pydcop.replication.path_utils import (
    Node,
    Path,
    PathsTable,
    cheapest_path_to,
    affordable_path_from,
    filter_missing_agents_paths,
    remove_path,
)

"""
UCS-based algorithm for replica distribution
for AAMAS 2018

This computation must be deployed on all agents in a resilient system.
It is used to distribute replicas of an agent's computations on other agents in
the neighborhood.

When distributing the replicas, this algorithm takes into account both the 
cost of the route (communication) between agents and the hosting cost for an 
(agent, computation) pair.

This algorithm is based on dict_ucs,  a distributed uniform-cost 
graph-exploring algorithm . See dict_ucs.py for a quick description of the 
principles of this algorithm. 

To account for hosting costs, we introduces for each 
agent' a' a virtual node in the graph (named __hosting__). The cost of the 
route between 'a' and __hosting__ is set to the hosting cost for 'a' (which 
depends on the computation). With this graph modification, we can simply use 
the graph exploring algorithm to explore the agents in increasing path cost 
order, with cost including the route and the hosting cost of the last agent 
in the route.



"""


def build_replication_computation(
    agent: Agent, discovery: Discovery
) -> MessagePassingComputation:
    """
    Parameters
    ----------
    agent: Agent
        the agent the computation is replicating for. it
        contains necessary info like hosting and route cost.
    discovery: Discovery

    Returns
    -------
    A computation object to distribute replicas
    """
    c = UCSReplication(agent, discovery)
    return c


# Used type alias for the name of agents, computations and replication
# computations, to avoid confusion in complex data structure and signatures.
ComputationName = str
AgentName = Node

# Give a little more priority to replication message (compared to dcop
# algorithm messages) as we must end replication before next event.
MSG_REPLICATION = 20


class UCSReplicateMessage(Message):
    """
    This message is sent by an agent to it's neighbors to asks them to host
    replicas
    """

    def __init__(
        self,
        rep_msg_type: str,
        budget: float,
        spent: float,
        rq_path: Path,
        paths: PathsTable,
        visited: List[AgentName],
        computation_def: ComputationDef,
        footprint: float,
        replica_count: int,
        hosts: List[str],
    ):
        super().__init__("ucs_replicate", None)
        # to to float computations, cannot check for strict >= 0
        assert budget >= -0.01
        assert spent >= -0.01
        self._rep_msg_type = rep_msg_type
        self._rq_path = rq_path
        self._computation_def = computation_def
        self._footprint = footprint
        self._budget = budget
        self._spent = spent
        self._replica_count = replica_count
        self._paths = paths
        self._visited = visited
        self._hosts = hosts

    @property
    def rep_msg_type(self):
        return self._rep_msg_type

    @property
    def rq_path(self) -> Path:
        return self._rq_path

    @property
    def computation_def(self) -> ComputationDef:
        return self._computation_def

    @property
    def footprint(self) -> float:
        return self._footprint

    @property
    def budget(self) -> float:
        return self._budget

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def replica_count(self) -> int:
        return self._replica_count

    @property
    def paths(self) -> PathsTable:
        return self._paths

    @property
    def visited(self) -> List[AgentName]:
        return self._visited

    @property
    def hosts(self) -> List[AgentName]:
        return self._hosts

    @property
    def size(self):
        table_size = sum(len(p) for _, p in self._paths)
        return len(self.hosts) + len(self._visited) + len(self._rq_path)

    def __str__(self):
        return "UCSReplicateMessage({}, {}, {}, {}, {})".format(
            self.computation_def.name,
            self.budget,
            self.spent,
            self.replica_count,
            self.rq_path,
        )

    def __repr__(self):
        return "UCSReplicateMessage({}, {}, {}, {}, {})".format(
            self.computation_def.name,
            self.budget,
            self.spent,
            self.replica_count,
            self.rq_path,
        )

    def __eq__(self, other):
        if type(other) != UCSReplicateMessage:
            return True
        if (
            self.computation_def == other.computation_def
            and self.budget == other.budget
            and self.hosts == other.hosts
        ):
            return True
        return False


def replication_computation_name(agt_name: AgentName) -> str:
    return "_replication_" + agt_name


class ReplicationTracker(object):
    """
    The ReplicationTracker is a simple container used to track the number of
    replication operations currently in progress for computations.
    """

    def __init__(self):
        self.replicating = {}

    def add(self, computations: Iterable[ComputationName]):
        for c in computations:
            count = self.replicating.get(c, 0)
            self.replicating[c] = count + 1

    def remove(self, computations: Iterable[ComputationName]):
        for computation in computations:
            self.replicating[computation] -= 1
        self.replicating = {
            comp: count for comp, count in self.replicating.items() if count > 0
        }

    def in_progress(self):
        return [c for c in self.replicating]

    def is_done(self, computation: ComputationName):
        return computation not in self.replicating

    def is_empty(self):
        return len(self.replicating) == 0

    def __repr__(self):
        return repr(self.replicating)


class UCSReplication(MessagePassingComputation):
    """
    This computation implements the d-ucs_replication algorithm.
    It must run on all agents in a resilient system.

    When the replication is over,

      * self._replica_hosts contains, for all the computations the current
        agent is responsible for, the list of other agents hosting a the
        replicas
      * self._hosted_replicas contains the list of replicas the current agent
        has accepted to host a replica of.

    Parameters
    ----------
    agent: Agent
        the agent the computation is replicating for. It is needed to access
        the lost of already hosted computations, hosting and route costs.
    discovery: Discovery
    logger: optional logger instance
        if not given an logger is automatically created with the name
        'ucs_replication.' + computation_name

    """

    def __init__(self, agent: Agent, discovery: Discovery, k_target=3, logger=None):
        super().__init__(replication_computation_name(agent.name))
        self.agent = agent
        self.agt_name = agent.name
        self.agent_def = agent.agent_def
        self.computations = (
            {}
        )  # type: Dict[ComputationName, Tuple[ComputationDef, float]]
        self.discovery = discovery
        self.k_target = k_target

        # Replicas hosted by this agent (with their footprint):
        self._hosted_replicas = (
            {}
        )  # type: Dict[ComputationName, Tuple[AgentName, float]]

        # Computation definitions for the replica hosted by this agent
        self.replicas = {}  # type: Dict[ComputationName, ComputationDef]

        # Hosts for the replicas of the computations this agent is
        self._replica_hosts = defaultdict(lambda: set())
        # type: Dict[ComputationName, Set[AgentName]]

        # Computation that are currently being replicated.
        self._replication_in_progress = ReplicationTracker()

        # Cache for the replication computations from other agents. These are
        # the computations we will communicate with to distribute our replicas.
        self._replication_computations_cache = (
            set()
        )  # type: Set[Tuple[AgentName, float]]

        self._pending_requests = {}
        self._removed_agents = set()

        self.logger = (
            logger
            if logger is not None
            else logging.getLogger("ucs_replication." + self.name)
        )

    @property
    def hosted_replicas(self) -> Dict[ComputationName, Tuple[AgentName, float]]:
        """
        List of hosted replica.

        Returns
        -------
        The list of hosted replica with their original agent and footprint.
        """
        return self._hosted_replicas

    def replication_neighbors(self) -> Set[AgentName]:
        """
        List of other neighbor replication computations that can be requested
        to host replicas.

        Returns
        -------
        Set[AgentName]
            A set of replication computations names.

        """
        # We use a cache to avoid re-computing all replication computations,
        # and re-registering them, every time.
        if not self._replication_computations_cache:

            # Find agents hosting the neighbor computations of our computations
            for c_def, _ in self.computations.values():
                for neighbor_name in c_def.node.neighbors:
                    if neighbor_name in self.computations:
                        continue
                    agt = self.discovery.computation_agent(neighbor_name)
                    self._replication_computations_cache.add((agt, self.route(agt)))
                    rep_comp = replication_computation_name(agt)
                    self.discovery.register_computation(rep_comp, agt, publish=False)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Potential path target: {self._replication_computations_cache}"
                )

        return self._replication_computations_cache

    def add_computation(self, comp_def: ComputationDef, footprint: float):
        """
        Add a new computation to be replicated.
        The replication process does not start automatically, `replicate(
        comp)` must be used for that purpose.

        Parameters
        ----------
        comp_def: ComputationDef
            the definition of the computation to replicate
        footprint: float
            the footprint of teh computation. It could be computed from the
            computationDef but it's expensive (requires importing the algo
            module) while the caller usually already have this value.

        Raises
        ------
        ValueError, if the computation is already registered
        to this replication computation.
        """
        comp_name = comp_def.node.name
        if comp_def.node.name in self.computations:
            raise ValueError("adding already present computation %s", comp_def)

        self.computations[comp_name] = comp_def, footprint
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"add computation {comp_name} to replicate")

    def remove_computation(self, computation: ComputationName):
        """
        Removes a computation.
        Parameters
        ----------
        computation: ComputationName
            The name of the computation

        """
        if computation not in self.computations:
            self.logger.warning(
                "Attempting to remove unknown computation %s", computation
            )
        else:
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"Removing computation to replicate {computation}")
            self.computations.pop(computation)

    def replicate(
        self,
        k_target: int,
        computations: Union[None, ComputationName, List[ComputationName]] = None,
    ):
        """
        Launch replication process for the computation(s) passed as argument.

        Parameters
        ----------
        k_target: int
            target number of replicas
        computations: None, string or list of strings.
            If computations is a string, is is considered as a single
            computation to be replicated/
            If it is a list of string, it is the list of computations
            that must be registered.
            If computations is None, all computations are registered.

        """
        if computations is None:
            self.logger.info(
                f"Request for replicating all computations {computations} - {k_target}"
            )
            computations = [c for c in self.computations]

        if not computations:
            self.logger.info(f"No computation to replicate for {self.name} ")
            self.replication_done(dict(deepcopy(self._replica_hosts)))
            return
        elif type(computations) == ComputationName:
            if computations not in self.computations:
                msg = "Requesting replication of unknown computation {}".format(
                    computations
                )
                self.logger.error(msg)
                raise ValueError(msg)
            computations = [computations]
        else:
            unknown = [c for c in computations if c not in self.computations]
            if unknown:
                msg = f"Requesting replication of unknown computation {unknown}"
                self.logger.error(msg)
                raise ValueError(msg)

        self._replication_in_progress.add(computations)
        neighbors = self.replication_neighbors()
        if not neighbors:
            self.logger.warning(
                f"Cannot replicate computations {computations} : no neighbor"
            )
            self.replication_done(dict(deepcopy(self._replica_hosts)))
            return

        self.logger.info(
            f"Starting replications of computations {computations} on neighbors {neighbors} - {k_target}"
        )

        for c in computations:
            # initialize paths with our neighbors and their costs
            paths = [(c, (self.agt_name, n)) for n, c in neighbors]
            paths.sort()
            budget = min(c for c, _ in paths)
            visited = [self.agt_name]
            comp_def, footprint = self.computations[c]
            self.on_replicate_request(
                budget,
                0,
                (self.agt_name,),
                paths,
                visited,
                comp_def,
                footprint,
                replica_count=k_target,
                hosts=[],
            )

    def on_start(self):
        # Register to all agents event, in order to use them for distribution
        #  or discard replica on removed agents
        self.discovery.subscribe_all_agents(self._on_agent_event)

    def on_stop(self):
        c_names = list(self.replicas.keys())
        self.logger.info(
            f"Stopping replication computation {self.name}, unregister hosted replicas {c_names}"
        )
        # The replication computation stops: unregister all replicas that
        # were hosted here:
        for c_name in c_names:
            self.remove_replica(c_name)

    @register("ucs_replicate")
    def _on_replicate_msg(self, sender_name: str, msg: UCSReplicateMessage, _: float):
        """
        This method is called when receiving a request from another agent to
        host a replica.

        Parameters
        ----------
        sender_name: computation name as a str
            the name of the computation that sent the message
        msg: UCSReplicateMessage
            the message
        _: float
            reception time, not used

        """
        if msg.rep_msg_type == "replicate_request":
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Received replication request from {sender_name}, {msg}"
                )
            self.on_replicate_request(
                msg.budget,
                msg.spent,
                msg.rq_path,
                msg.paths,
                msg.visited,
                msg.computation_def,
                msg.footprint,
                msg.replica_count,
                msg.hosts,
            )
        elif msg.rep_msg_type == "replicate_answer":
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Received replication answer from {sender_name}, {msg}"
                )
            agent = msg.rq_path[-1]  # last()
            pending = self._pending_requests.pop(
                (agent, msg.computation_def.name), False
            )
            if not pending:
                # If not in pending request : error !
                self.logger.warning(
                    f"Unexpected answer {agent}, {msg.computation_def.name} - "
                    f"{msg} not in {list(self._pending_requests.keys())}"
                )

            self.on_replicate_answer(
                msg.budget,
                msg.spent,
                msg.rq_path,
                msg.paths,
                msg.visited,
                msg.computation_def,
                msg.footprint,
                msg.replica_count,
                msg.hosts,
            )
        else:
            raise ValueError(f"Invalid message type {msg.rep_msg_type}")

    def on_replicate_request(
        self,
        budget: float,
        spent: float,
        rq_path: Path,
        paths: PathsTable,
        visited: List[AgentName],
        comp_def: ComputationDef,
        footprint: float,
        replica_count: int,
        hosts: List[str],
    ):
        assert self.agt_name == rq_path[-1]  # last()
        comp_name = comp_def.name
        remove_path(paths, rq_path)
        if self.agt_name not in visited:  # first visit for this node
            visited.append(self.agt_name)
            self._add_hosting_path(spent, comp_name, rq_path, paths)

        neighbors = self.replication_neighbors()

        # If some agents have left during replication, some may still be in
        # the received paths table even though sending replication_request to
        # them would block the replication.
        if self._removed_agents:
            paths = filter_missing_agents_paths(paths, self._removed_agents)

        # Available & affordable with current remaining budget, paths from here:
        # affordable_paths = affordable_path_from(rq_path, budget + spent, paths)
        # self.logger.debug(
        #     f"Affordable paths for {comp_name} on rq (b:{budget}, s:{spent}, {rq_path}) : "
        #     f": {affordable_paths} out of {paths}"
        # )

        # Paths to next candidates from paths table.
        target_paths = (
            rq_path + (p[0],)
            for p in affordable_path_from(rq_path, budget + spent, paths)
        )

        for target_path in target_paths:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Visiting for {comp_name} path {target_path}")
            forwarded, replica_count = self._visit_path(
                budget,
                spent,
                target_path,
                paths,
                visited,
                comp_def,
                footprint,
                replica_count,
                hosts,
            )
            if forwarded:
                return

        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                "No reachable path for %s with budget %s ", comp_name, budget
            )

        # Either:
        #  * No path : Not on a already known path
        #  * or all paths are too expensive for our current budget (meaning
        #    that we are at the last node in the known path)
        # In both cases, we now look at neighbors costs and store them if we
        # do not already known a cheaper path to them.
        neighbors_path = (
            (n, c, rq_path + (n,)) for n, c in neighbors if n not in visited
        )

        for n, r, p in neighbors_path:
            cheapest, cheapest_path = cheapest_path_to(n, paths)
            if cheapest > spent + r:
                remove_path(paths, cheapest_path)
                paths.append((spent + r, p))
                paths.sort()
            else:
                # self.logger.debug('Cheaper path known to %s : %s (%s)', p,
                #                   cheapest_path, cheapest)
                pass

        self._send_answer(
            budget,
            spent,
            rq_path,
            paths,
            visited,
            comp_def,
            footprint,
            replica_count,
            hosts,
        )

    def on_replicate_answer(
        self,
        budget: float,
        spent: float,
        rq_path: Path,
        paths: PathsTable,
        visited: List[AgentName],
        comp_def: ComputationDef,
        footprint: float,
        replica_count: int,
        hosts: List[str],
    ):
        *_, current, sender = rq_path
        comp_name = comp_def.name
        initial_path = rq_path[:-1]

        if self._removed_agents:
            paths = filter_missing_agents_paths(paths, self._removed_agents)

        # If all replica have been placed, report back to requester if any, or
        # signal that replication is done.
        if replica_count == 0:
            if len(rq_path) >= 3:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"All replica placed for {comp_name}, report back to requester"
                    )
                self._send_answer(
                    budget,
                    spent,
                    initial_path,
                    paths,
                    visited,
                    comp_def,
                    footprint,
                    replica_count,
                    hosts,
                )
                return
            else:
                self.computation_replicated(comp_name, hosts)
                return

        # If there are still replica to be placed, keep trying on neighbors
        back_path = rq_path[:-1]
        # affordable_paths = affordable_path_from(back_path, budget + spent, paths)

        # Paths to next candidates, avoiding the path we're coming from.
        target_paths = (
            back_path + (p[0],)
            for p in affordable_path_from(back_path, budget + spent, paths)
            if back_path + (p[0],) != rq_path
        )

        for target_path in target_paths:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Visiting for {comp_name} path {target_path}")
            forwarded, replica_count = self._visit_path(
                budget,
                spent,
                target_path,
                paths,
                visited,
                comp_def,
                footprint,
                replica_count,
                hosts,
            )
            if forwarded:
                return

        # Could not send to any neighbor: get back to requester
        if len(rq_path) >= 3:
            self._send_answer(
                budget,
                spent,
                initial_path,
                paths,
                visited,
                comp_def,
                footprint,
                replica_count,
                hosts,
            )
            return

        # no reachable candidate path and no ancestor to go back,
        # we are back at the start node: increase the budget
        if not paths:
            # Cannot increase budget, replica distribution is finished for
            # this computation, even if we have not reached target resiliency
            # level. Report the final replica distribution to the orchestrator.
            self.logger.warning(
                f"Could not reach target resiliency level for {comp_name}, "
                f"replicated on {hosts}"
            )
            self.computation_replicated(comp_name, hosts)
        else:
            budget = min(c for c, p in paths if p != rq_path)
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"Increase budget for computation {comp_name} : {budget}"
                )
            self.on_replicate_request(
                budget,
                0,
                (current,),
                paths,
                visited,
                comp_def,
                footprint,
                replica_count,
                hosts,
            )

    def _send_request(
        self,
        budget: float,
        spent: float,
        rq_path: Path,
        paths: PathsTable,
        visited: List[AgentName],
        comp_def: ComputationDef,
        footprint: float,
        replica_count: int,
        hosts: List[AgentName],
    ):
        target_agt = rq_path[-1]
        cost_to_next = self.route(target_agt)
        budget_to_next = budget - cost_to_next
        spent_to_next = spent + cost_to_next

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"sending replica request from {self.name} to {target_agt} "
                f"for {rq_path} - {comp_def.name} (b:{budget_to_next}, c: {cost_to_next})"
            )
        self.post_msg(
            replication_computation_name(target_agt),
            UCSReplicateMessage(
                "replicate_request",
                budget_to_next,
                spent_to_next,
                rq_path,
                paths,
                visited,
                comp_def,
                footprint,
                replica_count,
                hosts,
            ),
            MSG_REPLICATION,
        )

        # All request must be answered, otherwise the replication is stuck.
        # Keep track of all request sent.
        self._pending_requests[(target_agt, comp_def.name)] = (
            budget,
            spent,
            rq_path,
            paths.copy(),
            visited[:],
            comp_def,
            footprint,
            replica_count,
            hosts[:],
        )

    def _send_answer(
        self,
        budget: float,
        spent: float,
        rq_path: Path,
        paths: PathsTable,
        visited: List[AgentName],
        comp_def: ComputationDef,
        footprint: float,
        replica_count: int,
        hosts: List[AgentName],
    ):
        assert rq_path[-1] == self.agt_name
        target_agt = rq_path[-2]  # .before_last()
        cost_to_target = self.route(target_agt)
        budget += cost_to_target
        spent -= cost_to_target
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"sending replica answer from {self.name} to {target_agt} "
                f"for {comp_def.name} rq {rq_path} ({budget}, {spent}, {cost_to_target}) "
                f"paths : {paths}"
            )
        self.post_msg(
            replication_computation_name(target_agt),
            UCSReplicateMessage(
                "replicate_answer",
                budget,
                spent,
                rq_path,
                paths,
                visited,
                comp_def,
                footprint,
                replica_count,
                hosts,
            ),
            MSG_REPLICATION,
        )

    def route(self, target: AgentName) -> float:
        return self.agent_def.route(target)

    def footprint_comp(self, comp_name: ComputationName) -> float:
        return self.computations[comp_name][1]

    def computation_replicated(
        self, computation: ComputationName, hosts: List[AgentName]
    ):
        self._replication_in_progress.remove([computation])
        self._replica_hosts[computation].update(hosts)
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                f"Replica of {computation} accepted by {hosts}, "
                f"now replicated on {self._replica_hosts[computation]}, "
                f"is done {self._replication_in_progress.is_done(computation)}"
            )

        if self._replication_in_progress.is_empty():
            # All computations have been replicated, notify our agent.
            hosts = dict(deepcopy(self._replica_hosts))
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"All computations replicated : {hosts}")
            self.replication_done(hosts)
        else:
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    "Still waiting for replication of computations "
                    f"{self._replication_in_progress.in_progress()}"
                )

    def _on_agent_event(self, event: str, agent: str, _: Address):
        # On agent event, update the cache of replication computation and
        # restart replication when needed.
        agt_rep = replication_computation_name(agent)
        if event == "agent_removed":
            without = {
                (a, c) for a, c in self._replication_computations_cache if a != agent
            }
            if len(without) != len(self._replication_computations_cache):
                self._replication_computations_cache = without
                self._removed_agents.add(agent)

                # if we had pending request to this agent, we will never get an
                # answer
                self._answer_lost_requests(agent)

                # Re-launch replication for the computation(s) that have lost a
                # replica.
                self._replicate_on_agent_lost(agent)

        elif event == "agent_added":
            if agent != self.agt_name:
                # Only consider the agent if it is hosting a neighbor computation
                hosted = set(self.discovery.agent_computations(agent))
                if hosted.intersection(set(self.computations)):

                    self._replication_computations_cache.add((agent, self.route(agent)))
                    self.discovery.register_computation(agt_rep, agent, publish=False)

                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"Agent added {agent} , register replication computation {agt_rep}"
                        )

        else:
            self.logger.error("Unexpected agent event %s - %s ", event, agent)

    def replication_done(self, replica_hosts: Dict[ComputationName, Set[AgentName]]):
        # This method MUST be called when the distribution of the replicas
        # for a computation is finished. This is monitored by the
        # OrchestratedAgent and forwarded to the Orchestrator
        pass

    def remove_replica(self, computation: ComputationName):
        if computation not in self.replicas:
            self.logger.error("Cannot remove replica for %s : not hosted ", computation)
            return
        self.logger.info("Remove replica for %s ", computation)
        self.replicas.pop(computation)
        self._hosted_replicas.pop(computation)
        self.discovery.unregister_replica(computation, self.agt_name)

    def _add_hosting_path(
        self,
        spent: float,
        computation: ComputationName,
        rq_path: Path,
        paths: PathsTable,
    ):
        if computation not in self.computations:
            # Add a path to a virtual node with a route corresponding to the
            # hosting cost
            hosting_path = rq_path + ("__hosting__",)
            hosting_cost = spent + self.agent_def.hosting_cost(computation)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Add path to host %s on local hosting node %s " "with cost %s ",
                    computation,
                    hosting_path,
                    hosting_cost,
                )
            paths.append((hosting_cost, hosting_path))
            paths.sort()

    def _visit_path(
        self,
        budget: float,
        spent: float,
        target_path: Path,
        paths: PathsTable,
        visited: List[AgentName],
        comp_def: ComputationDef,
        footprint: float,
        replica_count: int,
        hosts: List[str],
    ):
        """
        Visit a path in the replication graph.

        Visiting can means attempting to host a replica, if we are on a
        __hosting__ node, or forwarding to another agent or answering the
        requester.

        Parameters
        ----------
        budget
        spent
        target_path
        paths
        visited
        comp_def
        footprint
        replica_count
        hosts

        Returns
        -------
        forwarded: boolean
            a boolean indicating if the request has been answered or
            forwarded to another agent.
        replica_count: int
            the updated replica count.
        """
        if target_path[-1] == "__hosting__":
            # We are actually 'visiting' the '__hosting__' virtual node
            # so we must remove it form the paths.
            remove_path(paths, target_path)
            if self._can_host(target_path[0], comp_def.name, footprint):
                self._accept_replica(target_path[0], comp_def, footprint)
                hosts.append(self.agent_def.name)
                replica_count -= 1
                if replica_count == 0:
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            "Target resiliency reached for %s, report back to "
                            "requester , hosts : %s",
                            comp_def.name,
                            hosts,
                        )
                    self._send_answer(
                        budget,
                        spent,
                        target_path[:-1],
                        paths,
                        visited,
                        comp_def,
                        footprint,
                        replica_count,
                        hosts,
                    )
                    return True, replica_count
                return False, replica_count
            # If the cheapest path was __hosting__, we can still try
            # to visit the next path (as we known __hosting__ never
            # have any other neighbor) => consider the request as not forwarded
            return False, replica_count

        self._send_request(
            budget,
            spent,
            target_path,
            paths,
            visited,
            comp_def,
            footprint,
            replica_count,
            hosts,
        )
        return True, replica_count

    def _replicate_on_agent_lost(self, agent: AgentName):
        """
        Re-launch replication for computation which had a replica hosted on
        a removed agent.

        Parameters
        ----------
        agent: AgentName
            the removed agent
        """
        removed_replicas = []
        # update list of valid replicas
        for replica, agents in self._replica_hosts.items():
            if agent in agents:
                removed_replicas.append(replica)
                agents.remove(agent)

        if removed_replicas:
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    "Agent removed %s , discard from our replica " "host %s - %s ",
                    agent,
                    removed_replicas,
                    dict(self._replica_hosts),
                )

            for removed_replica in removed_replicas:
                missing_replica = self.k_target - len(
                    self._replica_hosts[removed_replica]
                )
                # Note that missing replica might <= 0 if we had more
                # than the requested number of replica, which can happen
                # when two replications happened concurrently
                # when repairing replication on 2 agents removal
                if missing_replica < 0:
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            "No need to re-replicate %s still have enough "
                            "replicas: %s",
                            removed_replica,
                            self._replica_hosts[removed_replica],
                        )
                else:
                    try:
                        self.replicate(missing_replica, removed_replica)
                    except UnknownComputation :
                        # Avoid crashing if the computation is not known at the moment
                        # This can happen (and is perfectly valid) if we are currently
                        # repairing this computation.
                        pass

    def _answer_lost_requests(self, agent: AgentName):
        lost_rqs = [
            (rq_agt, rq_comp)
            for rq_agt, rq_comp in self._pending_requests
            if rq_agt == agent
        ]
        for rq in lost_rqs:
            self.logger.warning("Lost request %s : %s", rq, self._pending_requests[rq])
            rq_agt, rq_comp = rq
            # send a fake answer for pending request, to avoid blocking replication
            (
                budget,
                spent,
                rq_path,
                paths,
                visited,
                comp_def,
                footprint,
                replica_count,
                hosts,
            ) = self._pending_requests[rq]
            self.on_replicate_answer(
                budget,
                spent,
                rq_path,
                paths,
                visited,
                comp_def,
                footprint,
                replica_count,
                hosts,
            )

        pass

    def _can_host(
        self, agent: AgentName, computation: ComputationName, footprint: float
    ):
        """
        Evaluates if a replica could be hosted on this agent.

        In order to accept a replica, an agent has be sure that it would be
        able to activate it (with regards to its own capacity). As at most
        k_target agents could fail at the same time, we must make sure that
        all replicas hosted for any subset of k_target agents could be hosted
        at the same time.

        Parameters
        ----------
        agent: AgentName
            the name of the agent currently hosting the active computation
            corresponding to this replica
        computation: ComputationName
            the name of the replica we could host a computation of
        footprint: float
            the footprint of this computation

        Returns
        -------
        A boolean indicating if the replica can be accepted.
        """

        # Do not accept the same computation twice. We can get a request for
        # a computation we already have replica for when repairing a
        # replication. For example, when an agent hosting a replica leaves
        # the system, all the agents it was hosting replica for must repair
        # the replication of their computations.
        if computation in self._hosted_replicas:
            return False

        max_footprint = self._max_footprint() + footprint

        # max_footprint = self._worst_case_footprint(agent, computation, footprint)
        remaining_capacity = self._remaining_capacity()

        if remaining_capacity >= max_footprint:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Accept replica of %s from %s on %s", computation, agent, self.name
                )
            return True
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Reject replica %s (%s) from %s, remaining %s, need %s",
                    computation,
                    footprint,
                    self.agent_def.capacity,
                    remaining_capacity,
                    max_footprint,
                )
            return False

    def _accept_replica(self, origin_agt, comp_def: ComputationDef, footprint):
        comp_name = comp_def.name
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                "Accepting replica %s on %s from %s with footprint " "%s",
                comp_name,
                self.name,
                origin_agt,
                footprint,
            )
        self._hosted_replicas[comp_name] = (origin_agt, footprint)
        self.replicas[comp_name] = comp_def
        self.discovery.register_computation(comp_name, origin_agt, publish=False)
        self.discovery.register_replica(comp_name, self.agt_name)

    def _remaining_capacity(self) -> float:
        """The remaining capacity on the agent"""
        remaining = self.agent_def.capacity
        for hosted in self.agent.computations():
            if hasattr(hosted, "footprint"):
                remaining -= hosted.footprint()
        return remaining

    memoize_footprint = {}

    def _max_footprint(self):
        """
        Max footprint if k-1 agents disappear
        Returns
        -------

        """
        tentative_agents = set(a for a, f in self._hosted_replicas.values())

        max_agt = min(self.k_target - 1, len(tentative_agents))
        max_footprint = 0
        for selected in itertools.combinations(tentative_agents, max_agt):
            try:
                total_footprint = self.memoize_footprint[selected]
            except KeyError:

                total_footprint = sum(
                    f for a, f in self._hosted_replicas.values() if a in selected
                )
                self.memoize_footprint[selected] = total_footprint

            max_footprint = max(total_footprint, max_footprint)
        return max_footprint

    def _worst_case_footprint(
        self, agent: AgentName, computation: ComputationName, footprint: float
    ) -> float:
        # An agent accepts a replica if it is sure be able to accept it no
        # matter the set of k agents that disappear simultaneously.
        tentative_hosted = self._hosted_replicas.copy()
        tentative_hosted[computation] = (agent, footprint)
        tentative_agents = set(a for a, f in tentative_hosted.values())

        # Compute the max footprint of the replicas hosted for any subset
        # of k <= k_target owner agents.
        max_agt = min(self.k_target, len(tentative_agents))
        max_footprint = 0
        for selected in itertools.combinations(tentative_agents, max_agt):
            total_footprint = sum(
                f for a, f in tentative_hosted.values() if a in selected
            )
            max_footprint = max(total_footprint, max_footprint)
        return max_footprint
