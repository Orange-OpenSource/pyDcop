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
from pydcop.infrastructure.computations import MessagePassingComputation, \
    Message
from pydcop.infrastructure.discovery import Discovery, Address
from pydcop.replication.path_utils import Node, Path, PathsTable, \
    cheapest_path_to, affordable_path_from, filter_missing_agents_paths

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


def build_replication_computation(agent: Agent,
                                  discovery: Discovery) \
        -> MessagePassingComputation:
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
MSG_REPLICATION = 18


class UCSReplicateMessage(Message):
    """
    This message is sent by an agent to it's neighbors to asks them to host
    replicas
    """

    def __init__(self, rep_msg_type: str,
                 budget: float, spent: float,
                 rq_path: Path,
                 paths: PathsTable,
                 visited: List[AgentName],
                 computation_def: ComputationDef,
                 footprint: float,
                 replica_count: int,
                 hosts: List[str],
                 ):
        super().__init__('ucs_replicate', None)
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

    def __str__(self):
        return 'UCSReplicateMessage({}, {}, {}, {}, {})'.format(
            self.computation_def.name, self.budget, self.spent,
            self.replica_count, self.rq_path)

    def __repr__(self):
        return 'UCSReplicateMessage({}, {}, {}, {}, {})'.format(
            self.computation_def.name, self.budget, self.spent,
            self.replica_count, self.rq_path)

    def __eq__(self, other):
        if type(other) != UCSReplicateMessage:
            return True
        if self.computation_def == other.computation_def and self.budget == \
                other.budget and self.hosts == other.hosts:
            return True
        return False


def replication_computation_name(agt_name: AgentName) -> \
        str:
    return '_replication_' + agt_name


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
        self.replicating = {comp: count
                            for comp, count in self.replicating.items()
                            if count > 0}

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
    def __init__(self, agent: Agent,
                 discovery: Discovery,
                 k_target=3,
                 logger=None):
        super().__init__(replication_computation_name(agent.name))
        self._msg_handlers.update({
            'ucs_replicate': self._on_replicate_msg,
        })
        self.agent = agent
        self.agt_name = agent.name
        self.agent_def = agent.agent_def
        self.computations = \
            {}  # type: Dict[ComputationName, Tuple[ComputationDef, float]]
        self.discovery = discovery
        self.k_target = k_target

        # Replicas hosted by this agent (with their footprint):
        self._hosted_replicas =\
            {}  # type: Dict[ComputationName, Tuple[AgentName, float]]

        # Computation definitions for the replica hosted by this agent
        self.replicas = {}  # type: Dict[ComputationName, ComputationDef]

        # Hosts for the replicas of the computations this agent is
        self._replica_hosts = \
            defaultdict(lambda: set()) \
            # type: Dict[ComputationName, Set[AgentName]]

        # Computation that are currently being replicated.
        self._replication_in_progress = ReplicationTracker()

        # Cache for the replication computations from other agents. These are
        # the computations we will communicate with to distribute our replicas.
        self._replication_computations_cache = \
            set()  # type: Set[AgentName]

        self._pending_requests = {}

        self.logger = logger if logger is not None \
            else logging.getLogger('ucs_replication.'+self.name)

    @property
    def hosted_replicas(self) -> Dict[ComputationName, Tuple[AgentName, float]]:
        """
        List of hosted replica.

        Returns
        -------
        The list of hosted replica with their original agent and footprint.
        """
        return deepcopy(self._hosted_replicas)

    def replication_neighbors(self) -> Set[AgentName]:
        """
        List of other neighbor replication computations that can be requested
        to host replicas.

        Here, we consider we can communicate with any agent in the system,
        (as opposed to previous version - see git history - where only
        neighbor agents were considered).

        Returns
        -------
        Set[AgentName]
            A set of replication computations names.

        """
        # We use a cache to avoid re-computing all replication computations,
        # and re-registering them, every time.
        if not self._replication_computations_cache:
            # Build the cache of replication computation from other agents.
            for agt in self.discovery.agents():
                if agt == self.agt_name:
                    continue
                self._replication_computations_cache.add(agt)
                rep_comp = replication_computation_name(agt)
                self.discovery.register_computation(rep_comp, agt,
                                                    publish=False)
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
            raise ValueError('adding already present computation %s',
                             comp_def)

        self.computations[comp_name] = comp_def, footprint
        self.logger.info('add computation %s to replicate in neighborhood ',
                         comp_name)

    def remove_computation(self, computation: ComputationName):
        """
        Removes a computation.
        Parameters
        ----------
        computation: ComputationName
            The name of the computation

        """
        if computation not in self.computations:
            self.logger.warning('Attempting to remove unknown computation %s',
                                computation)
        else:
            self.logger.info('Removing computation to replicate %s',
                             computation)
            self.computations.pop(computation)

    def replicate(self, k_target: int,
                  computations: Union[None, ComputationName,
                                      List[ComputationName]]=None):
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
            self.logger.info('Request for replications of all computations %s -'
                             ' %s', computations, k_target)
            computations = [c for c in self.computations]
        elif not computations:
            self.logger.info('No computation to replicate for %s ', self.name)
            self.replication_done(dict(deepcopy(self._replica_hosts)))
            return
        elif type(computations) == ComputationName:
            if computations not in self.computations:
                msg = 'Requesting replication of unknown computation {}' \
                      .format(computations)
                self.logger.error(msg)
                raise ValueError(msg)
            computations = [computations]
        else:
            unknown = [c for c in computations if c not in self.computations]
            if unknown:
                msg = 'Requesting replication of unknown computation {}' \
                    .format(unknown)
                self.logger.error(msg)
                raise ValueError(msg)

        self._replication_in_progress.add(computations)
        neighbors = self.replication_neighbors()
        if not neighbors:
            self.logger.warning('Cannot replicate computations %s : no '
                                'neighbor', computations)
            self.replication_done(dict(deepcopy(self._replica_hosts)))
            return

        self.logger.info('Starting replications of computations %s on '
                         'neighbors %s - %s', computations, neighbors, k_target)

        for c in computations:
            # initialize paths with our neighbors and their costs
            paths = {Path(self.agt_name, n): self.route(n)
                     for n in neighbors}
            budget = min(c for c in paths.values())
            visited = [self.agt_name]
            comp_def, footprint = self.computations[c]
            self.on_replicate_request(
                budget, 0, Path(self.agt_name), paths, visited,
                comp_def, footprint, replica_count=k_target, hosts=[])

    def on_start(self):
        # Register to all agents event, in order to use them for distribution
        #  or discard replica on removed agents
        self.discovery.subscribe_all_agents(self._on_agent_event)

    def on_stop(self):
        c_names = list(self.replicas.keys())
        self.logger.info('Stopping replication computation %s, unregister '
                         'hosted replicas %s', self.name, c_names)
        # The replication computation stops: unregister all replicas that
        # were hosted here:
        for c_name in c_names:
            self.remove_replica(c_name)

    def _on_replicate_msg(self, sender_name: str, msg: UCSReplicateMessage,
                          _: float):
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
        if msg.rep_msg_type == 'replicate_request':
            self.logger.debug('Received replication request from %s, %s',
                              sender_name, msg)
            self.on_replicate_request(msg.budget, msg.spent,
                                      msg.rq_path, msg.paths, msg.visited,
                                      msg.computation_def, msg.footprint,
                                      msg.replica_count, msg.hosts)
        elif msg.rep_msg_type == 'replicate_answer':
            self.logger.debug('Received replication answer from %s, %s',
                              sender_name, msg)
            agent = msg.rq_path.last()
            pending = self._pending_requests.pop(
                (agent, msg.computation_def.name), False)
            if not pending:
                # If not in pending request : error !
                self.logger.error('Unexpected answer %s - %s  not in %s',
                                  (agent, msg.computation_def.name), msg,
                                  list(self._pending_requests.keys()))

            self.on_replicate_answer(msg.budget, msg.spent,
                                     msg.rq_path, msg.paths, msg.visited,
                                     msg.computation_def, msg.footprint,
                                     msg.replica_count, msg.hosts)
        else:
            raise ValueError('Invalid message type ' + str(msg.rep_msg_type))

    def on_replicate_request(self, budget: float, spent: float,
                             rq_path: Path, paths: PathsTable,
                             visited: List[AgentName],
                             comp_def: ComputationDef, footprint: float,
                             replica_count: int, hosts: List[str]):
        assert self.agt_name == rq_path.last()

        if rq_path in paths:
            paths.pop(rq_path)
        if self.agt_name not in visited:  # first visit for this node
            visited.append(self.agt_name)
            self._add_hosting_path(spent, comp_def.name, rq_path, paths)

        neighbors = self.replication_neighbors()

        # If some agents have left during replication, some may still be in
        # the received paths table even though sending replication_request to
        # them would block the replication.
        paths = filter_missing_agents_paths(paths, neighbors | {self.agt_name})
        # Available & affordable with current remaining budget, paths from here:
        affordable_paths = affordable_path_from(rq_path, budget + spent, paths)

        # self.logger.debug('Affordable path %s %s %s %s \n%s', budget, spent,
        #                   rq_path, affordable_paths, paths)

        # Paths to next candidates from paths table.
        target_paths = (rq_path + Path(p.head()) for _, p in affordable_paths)

        for target_path in target_paths:
            forwarded, replica_count = \
                self._visit_path(budget, spent, target_path, paths, visited,
                                 comp_def, footprint, replica_count, hosts)
            if forwarded:
                return

        self.logger.info('No reachable path for %s with budget %s ',
                         comp_def.name, budget)

        # Either:
        #  * No path : Not on a already known path
        #  * or all paths are too expensive for our current budget (meaning
        #    that we are at the last node in the known path)
        # In both cases, we now look at neighbors costs and store them if we
        # do not already known a cheaper path to them.
        neighbors_path = ((n, self.route(n), rq_path + Path(n))
                          for n in neighbors if n not in visited)

        for n, r, p in neighbors_path:
            cheapest, cheapest_path = cheapest_path_to(n, paths)
            if cheapest > spent + r:
                if cheapest_path in paths:
                    paths.pop(cheapest_path)
                paths[p] = spent + r
            else:
                # self.logger.debug('Cheaper path known to %s : %s (%s)', p,
                #                   cheapest_path, cheapest)
                pass

        self._send_answer(budget, spent, rq_path, paths, visited,
                          comp_def, footprint, replica_count,
                          hosts)

    def on_replicate_answer(self, budget: float, spent: float,
                            rq_path: Path, paths: PathsTable,
                            visited: List[AgentName],
                            comp_def: ComputationDef, footprint: float,
                            replica_count: int, hosts: List[str]):
        *_, current, sender = rq_path
        initial_path = rq_path[:-1]

        paths = filter_missing_agents_paths(
            paths, self.replication_neighbors() | {self.agt_name})

        # If all replica have been placed, report back to requester if any, or
        # signal that replication is done.
        if replica_count == 0:
            if len(rq_path) >= 3:
                self.logger.debug(
                    'All replica placed for %s, report back to requester',
                    comp_def.name)
                self._send_answer(budget, spent, initial_path, paths,
                                  visited, comp_def, footprint,
                                  replica_count, hosts)
                return
            else:
                self.computation_replicated(comp_def.name, hosts)
                return

        # If there are still replica to be placed, keep trying on neighbors
        back_path = rq_path[:-1]
        affordable_paths = affordable_path_from(back_path, budget + spent,
                                                paths)
        # Paths to next candidates, avoiding the path we're coming from.
        target_paths = (back_path + Path(p.head())
                        for _, p in affordable_paths
                        if back_path + Path(p.head()) != rq_path)

        for target_path in target_paths:
            forwarded, replica_count = \
                self._visit_path(budget, spent, target_path, paths, visited,
                                 comp_def, footprint, replica_count, hosts)
            if forwarded:
                return

        # Could not send to any neighbor: get back to requester
        if len(rq_path) >= 3:
            self._send_answer(budget, spent, initial_path, paths, visited,
                              comp_def, footprint, replica_count, hosts)
            return

        # no reachable candidate path and no ancestor to go back,
        # we are back at the start node: increase the budget
        if not paths:
            # Cannot increase budget, replica distribution is finished for
            # this computation, even if we have not reached target resiliency
            # level. Report the final replica distribution to the orchestrator.
            self.computation_replicated(comp_def.name, hosts)
        else:
            budget = min(c for p, c in paths.items() if p != rq_path)
            self.logger.info('Increase budget for computation %s : %s',
                             comp_def.name, budget)
            self.on_replicate_request(budget, 0, Path(current),
                                      paths, visited, comp_def, footprint,
                                      replica_count, hosts)

    def _send_request(self, budget: float, spent: float,
                      rq_path: Path, paths: PathsTable,
                      visited: List[AgentName],
                      comp_def: ComputationDef, footprint: float,
                      replica_count: int, hosts: List[AgentName]):
        target_agt = rq_path.last()
        cost_to_next = self.route(target_agt)
        budget_to_next = budget - cost_to_next
        spent_to_next = spent + cost_to_next

        self.logger.debug('sending replica request from  %s to %s for %s - %s ('
                          'budget = %s, cost to next %s)',
                          self.name, target_agt, rq_path, comp_def.name,
                          budget_to_next, cost_to_next)
        self.post_msg(
            replication_computation_name(target_agt),
            UCSReplicateMessage('replicate_request', budget_to_next,
                                spent_to_next, rq_path, paths, visited,
                                comp_def, footprint, replica_count, hosts),
            MSG_REPLICATION
        )

        # All request must be answered, otherwise the replication is stuck.
        # Keep track of all request sent.
        self._pending_requests[(target_agt, comp_def.name)] = \
            (budget, spent, rq_path, paths.copy(), visited[:], comp_def,
             footprint, replica_count, hosts[:])

    def _send_answer(self, budget: float, spent: float,
                     rq_path: Path, paths: PathsTable,
                     visited: List[AgentName], comp_def: ComputationDef,
                     footprint: float, replica_count: int,
                     hosts: List[AgentName]):
        assert rq_path.last() == self.agt_name
        target_agt = rq_path.before_last()
        cost_to_target = self.route(target_agt)
        budget += cost_to_target
        spent -= cost_to_target
        self.logger.debug('sending replica answer from %s to %s for %s %s'
                          '( %s %s %s )',
                          self.name, target_agt, rq_path, comp_def.name,
                          budget, spent, cost_to_target)
        self.post_msg(
            replication_computation_name(target_agt),
            UCSReplicateMessage('replicate_answer', budget, spent,
                                rq_path, paths, visited,
                                comp_def, footprint, replica_count, hosts),
            MSG_REPLICATION
        )

    def route(self, target: AgentName) -> float:
        return self.agent_def.route(target)

    def footprint_comp(self, comp_name: ComputationName) -> float:
        return self.computations[comp_name][1]

    def computation_replicated(self, computation: ComputationName,
                               hosts: List[AgentName]):
        self._replication_in_progress.remove([computation])
        self._replica_hosts[computation].update(hosts)
        self.logger.info('Replica of %s accepted by %s, now replicated on %s, '
                         'is done %s', computation, hosts,
                         self._replica_hosts[computation],
                         self._replication_in_progress.is_done(computation))

        if self._replication_in_progress.is_empty():
            # All computations have been replicated, notify our agent.
            hosts = dict(deepcopy(self._replica_hosts))
            self.logger.info('All computations replicated : %s', hosts)
            self.replication_done(hosts)
        else:
            self.logger.info('Still waiting for replication of computations '
                             '%s', self._replication_in_progress.in_progress())

    def _on_agent_event(self, event: str, agent: str, _: Address):
        # On agent event, update the cache of replication computation and
        # restart replication when needed.
        agt_rep = replication_computation_name(agent)
        if event == 'agent_removed':
            self._replication_computations_cache.remove(agent)

            # if we had pending request to this agent, we will never get an
            # answer
            self._answer_lost_requests(agent)

            # Re-launch replication for the computation(s) that have lost a
            # replica.
            self._replicate_on_agent_lost(agent)

        elif event == 'agent_added':
            self._replication_computations_cache.add(agent)
            self.discovery.register_computation(agt_rep, agent, publish=False)
            self.logger.info('Agent added %s , register replication '
                             'computation %s', agent, agt_rep)

        else:
            self.logger.error('Unexpected agent event %s - %s ', event, agent)

    def replication_done(self, replica_hosts: Dict[ComputationName,
                                                   Set[AgentName]]):
        # This method MUST be called when the distribution of the replicas
        # for a computation is finished. This is monitored by the
        # OrchestratedAgent and forwarded to the Orchestrator
        pass

    def remove_replica(self, computation: ComputationName):
        if computation not in self.replicas:
            self.logger.error('Cannot remove replica for %s : not hosted ',
                              computation)
            return
        self.logger.info('Remove replica for %s ', computation)
        self.replicas.pop(computation)
        self._hosted_replicas.pop(computation)
        self.discovery.unregister_replica(computation, self.agt_name)

    def _add_hosting_path(self, spent: float, computation: ComputationName,
                          rq_path: Path, paths: PathsTable):
        if computation not in self.computations:
            # Add a path to a virtual node with a route corresponding to the
            # hosting cost
            hosting_path = rq_path + Path('__hosting__')
            hosting_cost = spent + self.agent_def.hosting_cost(computation)
            self.logger.debug('Add path to host %s on local hosting node %s '
                              'with cost %s ', computation,
                              hosting_path, hosting_cost,)
            paths[hosting_path] = hosting_cost

    def _visit_path(self, budget: float, spent: float,
                    target_path: Path, paths: PathsTable,
                    visited: List[AgentName],
                    comp_def: ComputationDef, footprint: float,
                    replica_count: int, hosts: List[str]):
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
        if target_path.last() == '__hosting__':
            # We are actually 'visiting' the '__hosting__' virtual node
            # so we must remove it form the paths.
            paths.pop(target_path)
            if self._can_host(target_path.head(), comp_def.name, footprint):
                self._accept_replica(target_path.head(), comp_def, footprint)
                hosts.append(self.agent_def.name)
                replica_count -= 1
                if replica_count == 0:
                    self.logger.info(
                        'Target resiliency reached for %s, report back to '
                        'requester , hosts : %s', comp_def.name, hosts)
                    self._send_answer(budget, spent, target_path[:-1], paths,
                                      visited, comp_def, footprint,
                                      replica_count, hosts)
                    return True, replica_count
                return False, replica_count
            # If the cheapest path was __hosting__, we can still try
            # to visit the next path (as we known __hosting__ never
            # have any other neighbor) => consider the request as not forwarded
            return False, replica_count

        self._send_request(budget, spent, target_path, paths, visited,
                           comp_def, footprint, replica_count, hosts)
        return True, replica_count

    def _replicate_on_agent_lost(self, agent: AgentName):
        """
        Re-launch replication for computation which had a replica hosted on
        a removed agent
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
            self.logger.info('Agent removed %s , discard from our replica '
                             'host %s - %s ', agent, removed_replicas,
                             dict(self._replica_hosts))

            for removed_replica in removed_replicas:
                missing_replica = self.k_target - \
                                  len(self._replica_hosts[removed_replica])
                # Note that missing replica might <= 0 if we had more
                # than the requested number of replica, which can happen
                # when two replications happened concurrently
                # when repairing replication on 2 agents removal
                if missing_replica < 0:
                    self.logger.info(
                        'No need to re-replicate %s still have enough '
                        'replicas: %s',
                        removed_replica, self._replica_hosts[removed_replica])
                else:
                    self.replicate(missing_replica, removed_replica)

    def _answer_lost_requests(self, agent: AgentName):
        lost_rqs = [(rq_agt, rq_comp)
                    for rq_agt, rq_comp in self._pending_requests
                    if rq_agt == agent]
        for rq in lost_rqs:
            self.logger.warning('Lost request %s : %s', rq,
                                self._pending_requests[rq])
            # TODO: fake answer for pending request
            self._pending_requests.pop(rq)

        pass

    def _can_host(self, agent: AgentName, computation: ComputationName,
                  footprint: float):
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

        max_footprint = self._worst_case_footprint(agent, computation,
                                                   footprint)
        remaining_capacity = self._remaining_capacity()

        if remaining_capacity >= max_footprint:
            self.logger.debug('Accept replica of %s from %s on %s',
                              computation, agent, self.name)
            return True
        else:
            self.logger.debug(
                'Reject replica %s (%s) from %s, remaining %s, need %s',
                computation, footprint, self.agent_def.capacity,
                remaining_capacity, max_footprint)
            return False

    def _accept_replica(self, origin_agt, comp_def: ComputationDef, footprint):
        self.logger.info(
            'Accepting replica %s on %s from %s with footprint '
            '%s', comp_def.name, self.name, origin_agt,
            footprint)
        self._hosted_replicas[comp_def.name] = (origin_agt, footprint)
        self.replicas[comp_def.name] = comp_def
        self.discovery.register_computation(comp_def.name, origin_agt,
                                            publish=False)
        self.discovery.register_replica(comp_def.name, self.agt_name)

    def _remaining_capacity(self) -> float:
        """The remaining capacity on the agent"""
        remaining = self.agent_def.capacity
        for hosted in self.agent.computations():
            if hasattr(hosted, 'footprint'):
                remaining -= hosted.footprint()
        return remaining

    def _worst_case_footprint(self, agent: AgentName,
                              computation: ComputationName,
                              footprint: float) -> float:
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
            total_footprint = sum(f for a, f in tentative_hosted.values()
                                  if a in selected)
            max_footprint = max(total_footprint, max_footprint)
        return max_footprint
