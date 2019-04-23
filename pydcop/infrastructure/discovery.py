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
The discovery module provide an implementation for a discovery mechanism.
This mechanism allows any agent to discover other agents, their address and
and the computations they host.

The current implementation relies on a central directory, which will
generally be hosted on a central agent like an Orchestrator. However,
its interface has been designed so that it could be replaced by a
distributed implementation.

The discovery mecanism is made of two main components:

 * a single Directory
 * one Discovery instance for each agent

The directory must be manually deployed on an agent, like this:

    agt_dir = Agent('agt_dir', InProcessCommunicationLayer())
    directory = Directory(agt_dir.discovery)
    agt_dir.add_computation(directory.directory_computation)
    agt_dir.discovery.use_directory('agt_dir', agt_dir.address)
    agt_dir.start()
    agt_dir.run(directory.directory_computation.name)

When creating an agent, it is already provided with a discovery instance. The
address of the directory must be given using `use_directory`:

    agt1 = Agent('agt1', InProcessCommunicationLayer())
    agt1.discovery.use_directory('agt_dir', agt_dir.address)
    agt1.start()

The unit tests in `test_infra_discovery.py` gives many example of the
use of Discovery mechanism.

"""
import logging
from typing import Callable, List, Optional, Any, Dict, Tuple, Union

from collections import defaultdict
from typing import Set

from pydcop.infrastructure.computations import MessagePassingComputation, \
    Message, message_type

Address = Any
MSG_DISCOVERY = 5  # low value : discovery message have a very high priority

AgentName = str
ComputationName = str
DiscoveryName = str  # name of a discovery computation


class DiscoveryException(Exception):
    pass


class UnknownAgent(DiscoveryException):
    pass


class UnknownComputation(DiscoveryException):
    pass


PublishAgentMessage = message_type(
    'publish_agent', ['agents', 'address'])  # str, Address

UnPublishAgentMessage = message_type(
    'unpublish_agent', ['agent'])  # str

SubscribeAgentMessage = message_type(
    'subscribe_agent', ['agent', 'subscribe'])  # str, bool

PublishComputationMessage = message_type(
    'publish_computation', ['computation', 'agent', 'address'])

UnPublishComputationMessage = message_type(
    'unpublish_computation', ['computation', 'agent'])

SubscribeComputationMessage = message_type(
    'subscribe_computation', ['computation', 'subscribe'])  # str, bool

PublishReplicaMessage = message_type(
    'publish_replica', ['replica', 'agent', 'publish'])  # str, str, bool

SubscribeReplicaMessage = message_type(
    'subscribe_replica', ['replica', 'subscribe'])  # ComputationName, bool


class DirectoryComputation(MessagePassingComputation):
    """
    Computation for a centralized discovery service Directory.

    When using the Directory, this computation must be added to the agent
    that will be responsible for hosting the directory. See Directory doc
    for an example.

    See Also
    --------
    Directory, Discovery
    """

    def __init__(self, directory: 'Directory'):
        super().__init__('_directory')
        self.logger = logging.getLogger('pydcop.discovery.directory')
        self.directory = directory
        self._handlers = {
            'publish_agent': self._on_publish_agent,
            'unpublish_agent': self._on_unpublish_agent,
            'subscribe_agent': self._on_subscribe_agent,
            'publish_computation': self._on_publish_computation,
            'unpublish_computation': self._on_unpublish_computation,
            'subscribe_computation': self._on_subscribe_computation,
            'publish_replica': self._on_publish_replica,
            'subscribe_replica': self._on_subscribe_replica
        }

    @property
    def type(self) -> str:
        return 'directory'

    def on_start(self):
        # nothing to do on startup for directory
        pass

    def on_message(self, sender: DiscoveryName, msg: Message, t):
        self._handlers[msg.type](sender, msg)

    def _on_publish_agent(self, _: DiscoveryName, msg: PublishAgentMessage):
        self.logger.info('publication of agent %s with address %s',
                         msg.agents, msg.address)
        self.directory.register_agent(msg.agents, msg.address)

    def _on_unpublish_agent(self, _: DiscoveryName, msg: UnPublishAgentMessage):
        self.logger.info('un-publication of %s ', msg.agent)
        self.directory.unregister_agent(msg.agent)

    def _on_subscribe_agent(self, sender: DiscoveryName,
                            msg: SubscribeAgentMessage):
        if msg.subscribe:
            self.logger.info('Subscribe for agent %s from %s',
                             msg.agent, sender)
            if msg.agent == '*':
                self.directory.subscribe_all_agents(sender)
            else:
                try:
                    # automatically subscribe for this agent
                    self.directory.subscribe_to_agent(sender, msg.agent)
                    address = self.directory.agent_address(msg.agent)
                    # send answer
                    self.notify_agent_registered(sender, msg.agent, address)
                except UnknownAgent:
                    self.logger.warning('Unknown agent %s on lookup ',
                                        msg.agent)
        else:
            self.logger.info('UnSubscribe for agent %s from %s',
                             msg.agent, sender)
            self.directory.unsubscribe_from_agent(sender, msg.agent)

    def _on_publish_computation(self, _: DiscoveryName,
                                msg: PublishComputationMessage):
        self.logger.info('publication of computation %s hosted on %s',
                         msg.computation, msg.agent)
        self.directory.register_computation(msg.computation, msg.agent,
                                            msg.address)

    def _on_unpublish_computation(self, _: DiscoveryName,
                                  msg: UnPublishComputationMessage):
        self.logger.info('unpublication of computation %s ', msg.computation)
        self.directory.unregister_computation(msg.computation, msg.agent)

    def _on_subscribe_computation(self, sender: DiscoveryName,
                                  msg: SubscribeComputationMessage):
        if msg.subscribe:
            self.logger.info('Subscribe for computation %s from %s',
                             msg.computation, sender)
            agt = None
            try:
                # automatically subscribe for this agent
                self.directory.subscribe_to_computation(sender, msg.computation)
                agt = self.directory.computation_agent(msg.computation)
                address = self.directory.agent_address(agt)
                self.notify_computation_registered(
                    sender, msg.computation, agt, address)
            except UnknownComputation:
                # it is up to the requester to retry
                self.logger.warning('Unknown computation on lookup "%s"',
                                    msg.computation)
            except UnknownAgent:
                self.logger.warning('Unknown agent %s on lookup for '
                                    'computation %s', agt, msg.computation)
        else:
            self.logger.info('Unsubscribe for computation %s from %s',
                             msg.computation, sender)
            self.directory.unsubscribe_from_computation(sender, msg.computation)

    def _on_publish_replica(self, _: DiscoveryName,
                            msg: PublishReplicaMessage):
        if msg.publish:
            self.logger.info('publication of replica %s hosted on %s',
                             msg.replica, msg.agent)
            self.directory.register_replica(msg.replica, msg.agent)
        else:
            self.logger.info('un-publication of replica %s hosted on %s',
                             msg.replica, msg.agent)
            self.directory.unregister_replica(msg.replica, msg.agent)

    def _on_subscribe_replica(self, sender: DiscoveryName,
                              msg: SubscribeReplicaMessage):
        if msg.subscribe:
            self.logger.info('Subscribe for replica %s from %s',
                             msg.replica, sender)
            self.directory.subscribe_to_replicas(sender, msg.replica)
            try:
                for agt in self.directory.discovery.replica_agents(msg.replica):
                    self.notify_replica_registered(
                        sender, msg.replica, agt)
            except UnknownComputation:
                self.logger.warning('Subscriber for replicas of unknown '
                                    'commputation ' + msg.replica)
        else:
            self.logger.info('UnSubscribe for replica %s from %s',
                             msg.replica, sender)
            self.directory.unsubscribe_from_replicas(sender, msg.replica)

    def notify_agent_registered(self, interested: DiscoveryName,
                                agents: Union[AgentName, List[AgentName]],
                                address: Address) -> None:
        self.post_msg(interested, PublishAgentMessage(agents, address),
                      MSG_DISCOVERY)

    def notify_agent_unregistered(self, interested: DiscoveryName,
                                  agent: AgentName):
        self.post_msg(interested, UnPublishAgentMessage(agent),
                      MSG_DISCOVERY)

    def notify_computation_registered(
            self, interested: DiscoveryName,
            computation: ComputationName, agent: AgentName, address: Address):
        self.post_msg(interested,
                      PublishComputationMessage(computation, agent, address),
                      MSG_DISCOVERY)

    def notify_computation_unregistered(self, interested: DiscoveryName,
                                        computation: str, agent: AgentName):
        self.post_msg(interested,
                      UnPublishComputationMessage(computation, agent),
                      MSG_DISCOVERY)

    def notify_replica_registered(self, interested: DiscoveryName,
                                  replica: ComputationName, agent: AgentName):
        self.post_msg(interested,
                      PublishReplicaMessage(replica, agent, True),
                      MSG_DISCOVERY)

    def notify_replica_unregistered(self, interested: DiscoveryName,
                                    replica: ComputationName, agent: AgentName):
        self.post_msg(interested,
                      PublishReplicaMessage(replica, agent, False),
                      MSG_DISCOVERY)


class Directory(object):
    """
    Centralized implementation of a discovery mechanism.

    Notes
    -----
    When using the Directory, its computation must be added to the agent
    that will be responsible for hosting the directory. For example:

        agt_dir = Agent('agt_dir',InProcessCommunicationLayer())
        agt_dir.start()
        directory = Directory(agt_dir.discovery)
        agt_dir.add_computation(directory.directory_computation)
        agt_dir.discovery.use_directory('agt_dir', agt_dir.address)
        agt_dir.run(directory.directory_computation.name)


    Then, any agent must be given the agent (and it's address) hosting the
    directory:

        agt_dis = Agent('agt_dis', comm)
        agt_dis.discovery.use_directory('agt_dir', agt_dir.address)
        agt_dis.start()

    You generally never need to manipulate the directory directly, instead
    you may use the discovery instance given when building the directory.

    Parameters
    ----------
    discovery: a Discovery instance
        this instance will be updated with every registration and
        un-registration received by the directory. This must be the discovery
        instance of the agent hosting this directory (see example above).

    """

    def __init__(self, discovery: 'Discovery'):
        self.logger = logging.getLogger('pydcop.discovery.directory')
        self.directory_computation = DirectoryComputation(self)

        self._agents_data = {}  # type: Dict[AgentName, Address]
        self._computations_data = {}  # type: Dict[str,str]
        self.discovery = discovery

        # Subscription : message will be sent to interested agents
        self._subscription_agents = defaultdict(lambda: set()) \
            # type: Dict[AgentName, Set[DiscoveryName]]
        self._subscription_computations = defaultdict(lambda: set()) \
            # type: Dict[ComputationName, Set[DiscoveryName]]
        self._subscription_replicas = defaultdict(lambda: set()) \
            # type: Dict[ComputationName, Set[DiscoveryName]]
        # set of discovery_comp subscribed to all agents events.
        self._subscription_all_agents = set()  # type: Set[DiscoveryName]

        # local callback : these are directly called (no messages)
        self.on_register_agent = None  # type: Optional[Callable]
        self.on_unregister_agent = None  # type: Optional[Callable]
        self.on_register_computation = None  # type: Optional[Callable]
        self.on_unregister_computation = None  # type: Optional[Callable]

    def agent_address(self, agent: AgentName)-> Address:
        """
        Get an agent's address.

        Parameters
        ----------
        agent: str
            The name of the agent

        Returns
        -------
        Address
            The address of the agent

        Raises
        ------
        UnknownAgent
            If there is no knwon agent with this name.

        """
        try:
            return self._agents_data[agent]
        except KeyError:
            raise UnknownAgent(agent)

    def register_agent(self, agent: AgentName, address: Address):
        self.logger.debug('Registering agent %s on %s', agent, address)
        self._agents_data[agent] = address
        self.discovery.register_agent(agent, address, publish=False)
        for interested in self._subscription_agents[agent]:
            self.directory_computation.notify_agent_registered(
                interested, agent, address)
        for interested in self._subscription_all_agents:
            self.directory_computation.notify_agent_registered(
                interested, agent, address)

        if self.on_register_agent is not None:
            self.on_register_agent(agent, address)

    def unregister_agent(self, agent: AgentName):
        try:
            # agent un-registration is only allowed if the agent has no
            # non-technical computation registered
            non_technical = self.discovery.agent_computations(agent)
            if non_technical:
                raise DiscoveryException(
                    'Cannot unregister agent with non-technical '
                    'computations : {} - {} '.format(agent, non_technical))

            for computation in self.discovery.agent_computations(
                    agent, include_technical=True):
                self.unregister_computation(computation)
            self._agents_data.pop(agent)
            self.discovery.unregister_agent(agent, publish=False)
        except KeyError as ke:
            self.logger.error('KeyError in unregister_agent: %s', ke)
            return

        # cleanup any subscriptions for this removed agent, that is to say
        # the subscription this agent had (as it will not be able to receive
        # any notification now)
        self.logger.debug('Subscription cleanup on removing %s ', agent)
        agent_discovery = '_discovery_' + agent
        removed_subscriptions = []
        for target, subscriptions in self._subscription_agents.items():
            if agent_discovery in subscriptions:
                subscriptions.remove(agent_discovery)
                removed_subscriptions.append(target)
                self._subscription_agents[target] = subscriptions
        self.logger.debug('Removing agt subscription to %s for removed '
                          'agent %s', removed_subscriptions, agent)
        for target, subscriptions in self._subscription_computations.items():
            if agent_discovery in subscriptions:
                self.logger.debug('Removing comp subscription to %s for '
                                  'removed agent %s', target, agent)
                subscriptions.remove(agent_discovery)
                self._subscription_agents[target] = subscriptions
        try:
            self._subscription_all_agents.remove(agent_discovery)
        except KeyError:
            pass

        # notify interested agents
        interested_agents = self._subscription_agents[agent] | \
            self._subscription_all_agents
        self.logger.debug('notifying register agent for removal of %s : %s',
                          agent, interested_agents)
        for interested in interested_agents:
            self.directory_computation.notify_agent_unregistered(
                interested, agent)

        if self.on_unregister_agent is not None:
            self.on_unregister_agent(agent)

    def subscribe_all_agents(self, subscriber: DiscoveryName):
        self.logger.debug('Subscribing to all agents from %s', subscriber)
        self._subscription_all_agents.add(subscriber)

        # notify for an already registered agents
        all_agents = self.discovery.agents()
        all_addr = [self.discovery.agent_address(agt)
                    for agt in all_agents]
        self.logger.debug('notifying for already registered agents %s',
                          all_agents)
        for interested in self._subscription_all_agents:
            self.directory_computation.notify_agent_registered(
                interested, all_agents, all_addr)

    def subscribe_to_agent(self, subscriber: DiscoveryName, agent: AgentName):
        self._subscription_agents[agent].add(subscriber)

    def unsubscribe_from_agent(self, subscriber: DiscoveryName,
                               agent: AgentName):
        try:
            self._subscription_agents[agent].remove(subscriber)
        except KeyError:
            self.logger.warning('Un-subscribing from an unknown agent %s - %s',
                                agent, subscriber)
        except ValueError:
            self.logger.warning('Unknown subscriber %s when un-subscribing '
                                'from an agent %s ', subscriber, agent)

    def computation_agent(self, computation: ComputationName)-> AgentName:
        try:
            return self._computations_data[computation]
        except KeyError:
            raise UnknownComputation(computation)

    def register_computation(self, computation: ComputationName,
                             agent: AgentName,
                             address: Address=None):
        self._computations_data[computation] = agent
        self.discovery.register_computation(
            computation, agent, address, publish=False)
        address = address if address is not None else self._agents_data[agent]
        for interested in self._subscription_computations[computation]:
            self.directory_computation.notify_computation_registered(
                interested, computation, agent, address)
        if self.on_register_computation is not None:
            self.on_register_computation(agent, computation)

    def unregister_computation(self, computation: ComputationName,
                               agent: AgentName=None):
        try:
            self._computations_data.pop(computation)
            self.discovery.unregister_computation(computation)
        except (KeyError, UnknownComputation):
            return
        # notify interested agents
        for interested in self._subscription_computations[computation]:
            self.directory_computation.notify_computation_unregistered(
                interested, computation, agent)
        if self.on_unregister_computation is not None:
            self.on_unregister_computation(computation)

    def subscribe_to_computation(self, subscriber: DiscoveryName,
                                 computation: ComputationName):
        self._subscription_computations[computation].add(subscriber)

    def unsubscribe_from_computation(self, subscriber: DiscoveryName,
                                     computation: ComputationName):
        try:
            self._subscription_computations[computation].remove(subscriber)
        except KeyError:
            self.logger.warning('Un-subscribing from an unknown computation '
                                '%s - %s', computation, subscriber)
        except ValueError:
            self.logger.warning('Unknown subscriber %s when un-subscribing from'
                                ' an computation %s ', subscriber, computation)

    def register_replica(self, replica: ComputationName, agent: AgentName):
        self.discovery.register_replica(
            replica, agent, publish=False)
        for interested in self._subscription_replicas[replica]:
            self.directory_computation.notify_replica_registered(
                interested, replica, agent)

    def unregister_replica(self, replica: ComputationName, agent: AgentName):
        try:
            self.discovery.unregister_replica(replica, agent)
        except (KeyError, UnknownComputation):
            return
        # notify interested agents
        for interested in self._subscription_replicas[replica]:
            self.directory_computation.notify_replica_unregistered(
                interested, replica, agent)

    def subscribe_to_replicas(self, subscriber: DiscoveryName,
                              replica: ComputationName):
        self._subscription_replicas[replica].add(subscriber)

    def unsubscribe_from_replicas(self, subscriber: DiscoveryName,
                                  replica: ComputationName):
        try:
            self._subscription_replicas[replica].remove(subscriber)
        except KeyError:
            self.logger.warning('Un-subscribing from an unknown replica '
                                '%s - %s', replica, subscriber)
        except ValueError:
            self.logger.warning('Unknown subscriber %s when un-subscribing from'
                                ' an replica %s ', subscriber, replica)


class DiscoveryComputation(MessagePassingComputation):
    """
    Message passing computation for discovery.

    Parameters
    ----------
    agent_name: str
    discovery: Discovery
    """

    def __init__(self, agent_name: AgentName, discovery: 'Discovery'):
        super().__init__('_discovery_' + agent_name)
        self.discovery = discovery
        self.directory_name = None

        self.logger = discovery.logger
        self._handlers = {
            'publish_agent': self._on_agent_added,
            'unpublish_agent': self._on_agent_removed,
            'publish_computation': self._on_computation_added,
            'unpublish_computation': self._on_computation_removed,
            'publish_replica': self._on_replica_publish,
        }

    @property
    def type(self):
        return 'discovery'

    def on_start(self):
        pass

    def on_message(self, sender: DiscoveryName, msg: Message, t):
        self._handlers[msg.type](sender, msg)

    def _on_agent_added(self, _: DiscoveryName, msg: PublishAgentMessage):
        if isinstance(msg.agents, str):
            self.discovery.register_agent(msg.agents, msg.address,
                                          publish=False)
        else:
            for agent, address in zip(msg.agents, msg.address):
                self.discovery.register_agent(agent, address, publish=False)

    def _on_agent_removed(self, _: DiscoveryName, msg: UnPublishAgentMessage):
        self.discovery.unregister_agent(msg.agent, publish=False)

    def _on_computation_added(self, _: DiscoveryName,
                              msg: PublishComputationMessage):
        self.discovery.register_computation(msg.computation, msg.agent,
                                            msg.address, publish=False)

    def _on_computation_removed(self, _: DiscoveryName,
                                msg: UnPublishComputationMessage):
        self.discovery.unregister_computation(
            msg.computation, msg.agent, publish=False)
        pass

    def _on_replica_publish(self, _, msg: PublishReplicaMessage):
        if msg.publish:
            self.discovery.register_replica(msg.replica, msg.agent,
                                            publish=False)
        else:
            self.discovery.unregister_replica(msg.replica, msg.agent,
                                              publish=False)

    def send_to_directory(self, msg):
        if self.directory_name is not None:
            self.post_msg(self.directory_name, msg, MSG_DISCOVERY,
                          on_error='fail')

    def __str__(self):
        return 'DiscoveryComputation({})'.format(self.name)


def _is_technical(computation: str) -> bool:
    """
    Check if a computation is a technical computation.
    Parameters
    ----------
    computation: str
        a computation name.

    Returns
    -------
    True if it is a technical computation, false otherwise.
    """
    if computation.startswith('_'):
        return True
    if computation.startswith('B'):
        # filter reparation computation, like Bc004_026_a012
        return True
    return False


DiscoveryCallBack = Callable[[str, str, str], None]
CbRegistration = Tuple[DiscoveryCallBack, bool]


class Discovery(object):
    """
    A Discovery instance is used to keep track of agents and computations.

    Notes
    -----
    Discovery uses a specific computation to send and receive discovery
    messages to and from the directory. You generally do not need to create
    instance of Discovery manually : every agent is automatically provided
    with a discovery instance. However, you must provide the address of the
    directory to the agent:

        agt_dis.discovery.use_directory('agt_dir', agt_dir.address)

    If no directory is given, all discovery will stay local to the agent.

    Parameters
    ----------
    agent_name: str
        the name of the agent that will use the discovery instance.
    address: Address
        the address of the agent using this discovery instance. This is
        mandatory for this Discovery instance to be able to receive message
        from the directory.
    """
    def __init__(self, agent_name: AgentName, address: Address):
        super().__init__()
        self.own_agent = agent_name
        self.logger = logging.getLogger('pydcop.discovery.'+agent_name)

        # agent_name -> agent_address
        self._agents_data = {}  # type: Dict[AgentName, Address]
        # computation_name -> agent_name
        self._computations_data = {}  # type: Dict[ComputationName, AgentName]
        self._replicas_data = defaultdict(lambda: set()) \
            # type: Dict[ComputationName, Set[AgentName]]

        self._computation_cbs = defaultdict(lambda: []) \
            # type: Dict[ComputationName, List[CbRegistration]]
        self._agent_cbs = defaultdict(lambda: []) \
            # type: Dict[AgentName, List[CbRegistration]]
        self._replicas_cbs = defaultdict(lambda: []) \
            # type: Dict[ComputationName, List[CbRegistration]]
        self._all_agents_cbs = []  # type List[DiscoveryName]

        self.directory_ref = None, None, None
        self.discovery_computation = DiscoveryComputation(agent_name, self)
        # In order to be able to send any registration to the directory,
        # we must at least register the discovery locally, otherwise the
        # messaging system will not recognize the message as valid.
        self.register_computation(self.discovery_computation.name,
                                  agent_name, address, publish=False)

    def use_directory(self, agent: AgentName, address: Address):
        """
        Set the address of the directory to be used by this Discovery instance.

        This mus be called before starting the agent holding this discovery
        instance.

        Parameters
        ----------
        agent: str
            The name of the agent hosting the directory.
        address: Address
            The address of the agent hosting the directory.
        """
        directory_computation = '_directory'
        self.directory_ref = (agent, address, directory_computation)
        self.discovery_computation.directory_name = directory_computation
        self.register_agent(agent, address, publish=False)
        self.register_computation(directory_computation, agent, publish=False)

    def agents(self, filter_orchestrator=True) -> List[str]:
        """
        The list of registered agents.

        Parameters
        ----------
        filter_orchestrator: bool
            If False, the orchestrator is filtered out.

        Returns
        -------
        List[str]:
            A list of agents names.
        """
        if filter_orchestrator:
            return list(a for a in self._agents_data if a != 'orchestrator')
        else:
            return list(self._agents_data)

    def agent_address(self, agent: AgentName) -> Address:
        """
        Get an agent's address.

        Parameters
        ----------
        agent: str
            The name of the agent

        Returns
        -------
        Address
            The address of the agent

        Raises
        ------
        UnknownAgent
            If there is no knwon agent with this name.
        """
        try:
            return self._agents_data[agent]
        except KeyError:
            raise UnknownAgent('Unknown agent ' + str(agent))

    def register_agent(self, agent: AgentName, address: Address,
                       publish: bool=True):
        """
        Registers a an agent.

        Callback for this agent are fired, if any.

        Parameters
        ----------
        agent: str
            The name of the agent
        address: Address
            The address of the agent
        publish: bool
            If true, the registration will be sent to the directory,
            otherwise the agent will only be registered locally.

        """
        is_change = address != self._agents_data.get(agent, None)

        self._agents_data[agent] = address
        if publish:
            self.logger.info('Publishing agent %s with address %s ',
                             agent, address)
            msg = PublishAgentMessage(agent, address)
            self.discovery_computation.send_to_directory(msg)
        else:
            # too verbose, even for debug. Only uncomment for hairy
            # debugging sessions ...
            # self.logger.debug('Register locally agent %s with address %s ',
            #                   agent, address)
            pass

        # Fire agent-specific callbacks if any and if there was an actual change
        if not is_change:
            return

        if agent in self._agent_cbs:
            for cb, one_shot in self._agent_cbs[agent]:
                self.logger.debug('fire agent_added call back for %s : %s',
                                  agent, cb)
                cb('agent_added', agent, address)
            # Remove all one-shot callback for this computation
            self._agent_cbs[agent][:] = \
                [(cb, oneshot)
                 for cb, oneshot in self._agent_cbs[agent]
                 if not oneshot]
        for cb in self._all_agents_cbs:
            cb('agent_added', agent, address)

    def unregister_agent(self, agent: AgentName, publish: bool=True):
        """
        Un-registers an agent.

        Callback for this agent are fired, if any.
        Any computation must be un-registered before un-registering the agent.


        Parameters
        ----------
        agent: str
            the name of the agent
        publish: bool
            If true, the un-registration will be sent to the directory,
            otherwise the agent will only be un-registered locally.

        Raises
        ------
        DiscoveryException:
            If the are still some (non-technical) computation registred for
            this agent.
        """
        try:
            agent_computations = self.agent_computations(agent)
            if agent_computations:
                if publish:
                    raise DiscoveryException(
                        'Cannot unregister agent which has registered '
                        'computation: {} - {}'.format(agent,
                                                      agent_computations))
                else:
                    for c in agent_computations:
                        self.unregister_computation(c, agent, publish=False)

            self._agents_data.pop(agent)
            if publish:
                self.logger.info('Unregister agent %s', agent)
                self.discovery_computation.send_to_directory(
                    UnPublishAgentMessage(agent))
            else:
                self.logger.debug('Unregister locally agent %s', agent)

            # Fire callbacks if any
            if agent in self._agent_cbs:
                for cb, oneshot in self._agent_cbs[agent]:

                    self.logger.debug('fire agent_removed call back for %s',
                                      agent)
                    cb('agent_removed', agent, None)
                    if oneshot:
                        self.logger.debug('discard one-shot call back for '
                                          'agent %s', agent)
                        self._agent_cbs[agent].remove((cb, oneshot))

            for cb in self._all_agents_cbs:
                cb('agent_removed', agent, None)

        except KeyError:
            self.logger.info('Unknown agent %s , on unregister', agent)

    def subscribe_agent(self, agent: AgentName,
                        cb: Optional[DiscoveryCallBack]=None,
                        one_shot: bool=False)\
            -> DiscoveryCallBack:
        """
        Subscribe to an agent on the directory.

        When subscribed to an agent, the discovery instance is will be
        notified of any registration or un-registration for this agent.
        Subscribing also force this discovery instance to perform a lookup on
        the directory to find the address of the agent.

        The agent may or may not be registered on the directory. If it is
        already registered, the discovery instance will notified immediately,
        if it is not registered yet, this discovery instance will be
        notified (and updated) once the agent is registered (which may never
        happen).

        If a callback is provided, it will be called when receiving such
        notification, otherwise the local discovery will simply be updated.

        The callback must be a callable accepting 3 strings as parameters. The
        first parameter will be 'agent_added' or 'agent_removed' and the
        second and third parameter will be the agent's name and address.

        Parameters
        ----------
        agent: str
            The name of the agent
        cb: DiscoveryCallBack
            An optional callback as a callable accepting 3 strings as
            parameters.
        one_shot: bool
            If true, the callback will be discarded after one call. Notice
            that the discovery instance will still be updated by the
            directory for this agent.

        Returns
        -------
        Callable
            the callback, or None if no callback was given.

        See Also
        --------
        Discovery.unsubscribe_agent
        """
        self.logger.debug('Subscribe to agent %s : %s, %s',
                          agent, cb, one_shot)
        already_subscribed = agent in self._agent_cbs
        if cb is not None:
            self._agent_cbs[agent].append((cb, one_shot))
        if not already_subscribed or cb is None:
            self.discovery_computation.send_to_directory(
                SubscribeAgentMessage(agent, True))
        return cb

    def subscribe_all_agents(self, cb: Optional[Callable]=None):
        self.logger.debug('Subscribe to all agents events')
        if not self._all_agents_cbs:
            self.logger.debug('send all subscription to directory')
            self.discovery_computation.send_to_directory(
                SubscribeAgentMessage('*', True))
        else:
            self.logger.debug('NOt send all subscription to directory : %s',
                              self._all_agents_cbs)

        if cb is not None:
            self._all_agents_cbs.append(cb)
        return cb

    def unsubscribe_agent(self, agent: AgentName,
                          cb: Optional[DiscoveryCallBack]=None)-> int:
        """
        Cancel subscription for an agent.

        Removes the callback `cb` for the agent, or all callbacks if `cb` is
        None. If there is no callback left for this agent, the discovery
        instance subscription on the directory is removed for this agent.

        Parameters
        ----------
        agent: str
            The agent name
        cb: DiscoveryCallBack or None
            A callback that was previously registered for this agent with
            `subscribe_agent`. If None, all callbacks will be removed for
            this agent and the discovery instance will be un-subscribed from
            the directory for this agent.

        Returns
        -------
        int
            The number of callbacks removed.

        Raises
        ------
        ValueError
            If the callback is not registered for this agent.

        See Also
        --------
        Discovery.subscribe_agent
        """
        self.logger.debug('Un-subscribe from agent %s : %s',
                          agent, cb)
        removed = 0
        if agent in self._agent_cbs:
            for r_cb, one_shot in self._agent_cbs[agent][:]:
                if cb is None or r_cb == cb:
                    self._agent_cbs[agent].remove((r_cb, one_shot))
                    removed += 1
            if removed:
                if not self._agent_cbs[agent]:
                    self._agent_cbs.pop(agent)
                    self.logger.debug('No callback left for %s, unsubscribe '
                                      'on directory', agent)
                    self.discovery_computation.send_to_directory(
                        SubscribeAgentMessage(agent, False))
            elif cb is not None:
                raise ValueError(
                    'No corresponding callback found for agt %s : %s',
                    agent, cb)
        else:
            self.discovery_computation.send_to_directory(
                SubscribeAgentMessage(agent, False))
        return removed

    def computations(self, include_technical=False)-> List[str]:
        """
        List of computations.

        Notes
        -----
        Filtering is based on the method `__is_technical(name)`

        Parameters
        ----------
        include_technical: boolean
            If true, technical computations (e.g. directory, discovery,
            management) are included, otherwise only computations from the
            problem DCOP are returned.

        Returns
        -------
        List[str]
            The list of the name of all computations currently known by
            this discovery instance.
        """
        if include_technical:
            return list(self._computations_data)
        else:
            return list(c for c in self._computations_data
                        if not _is_technical(c))

    def computation_agent(self, computation: ComputationName)-> AgentName:
        """
        The agent hosting a computation

        Parameters
        ----------
        computation: str
            The name of the computation.

        Returns
        -------
        The name of the agent hosting this computation.

        Raises
        ------
        UnknownComputation if there is no known computation with this name
        """
        try:
            return self._computations_data[computation]
        except KeyError:
            raise UnknownComputation(computation)

    def agent_computations(self, agent: AgentName,
                           include_technical=False)-> List[str]:
        """
        List of computations hosted on an agent.

        Parameters
        ----------
        agent: str
            The agent name
        include_technical: bool
            If true, technical computations (e.g. directory, discovery,
            management) are included, otherwise only computations from the
            problem DCOP are returned.

        Returns
        -------
        List[str]
            The list of names of the computations hosted by the agent known by
            this discovery.
        """
        computations = []
        for c, a in self._computations_data.items():
            if agent == a:
                if include_technical or not _is_technical(c):
                    computations.append(c)
        return computations

    def register_computation(self, computation: ComputationName,
                             agent: Optional[AgentName]=None,
                             address: Optional[Address]=None,
                             publish: bool=True):
        """
        Registers a computation hosted on an agent.

        Parameters
        ----------
        computation: str
            The name of the computation.
        agent: Optional[str]
            The name of the agent hosting this computation. If not given,
            it is assumed that the computation is the agent using this
            discovery instance (and whose name was given when creating the
            instance)
        address: Optional[Address]
            If None, the address must be already known to this discovery
            instance, otherwise
        publish: bool
            If True, the computation registration will be published on the
            directory.

        Raises
        ------
        UnknownAgent
            If the address is not given and the agent is not registered on this
            discovery instance.
        """
        agent = self.own_agent if agent is None else agent

        # If the address is not provided, the agent must be already known.
        if address is None:
            self.agent_address(agent)

        is_change = agent != self._computations_data.get(computation, None)

        self._computations_data[computation] = agent
        # If we do not known this agent and the address is given, register it
        # at the same time.
        if address is not None:
            if agent not in self._agents_data:
                self.register_agent(agent, address, publish=False)
        if publish:
            self.logger.info('Publishing computation %s hosted on %s',
                             computation, agent)
            self.discovery_computation.send_to_directory(
                PublishComputationMessage(computation, agent, address))
        else:
            pass
            # self.logger.debug('Register locally computation %s hosted on %s',
            #                   computation, agent)

        # Fire callbacks only if there was an actual change
        if not is_change:
            return
        if computation in self._computation_cbs:
            for cb, oneshot in self._computation_cbs[computation]:
                self.logger.debug('fire computation_added call back for %s : '
                                  '%s', computation, cb)
                cb('computation_added', computation, agent)

            # Remove all one-shot callback for this computation
            self._computation_cbs[computation][:] = \
                [(cb, oneshot)
                 for cb, oneshot in self._computation_cbs[computation]
                 if not oneshot]

    def unregister_computation(self, computation: ComputationName,
                               agent: AgentName=None, publish: bool=True):
        """
        Un-registers a computation.

        Notes
        -----
        If the computation is unknown not exception is logged (& info is logged)

        Parameters
        ----------
        computation: str
            The computation name
        agent: str
            Optional, the name of the agent that hosts the computation. This
            has no effect except performing an additional check and is meant as
            a safety net.
        publish: bool
            If True, the computation un-registration will be published on the
            directory.

        Raises
        ------
        ValueError
            If agent is not None and does not match the agent this
            computation is registered with.
        """
        try:
            known_agent = self._computations_data[computation]
            if agent is not None and known_agent != agent:
                raise ValueError('Computation {} is known to be hosted on {}, '
                                 'not {} '.format(computation, known_agent,
                                                  agent))

            # Fire callbacks if any.
            # We must fire callback before unsubscribing, as unsubscribing
            # removes callbacks!
            if computation in self._computation_cbs:
                for cb, _ in self._computation_cbs[computation]:
                    self.logger.debug('fire computation_removed call back '
                                      'for %s', computation)
                    cb('computation_removed', computation, agent)

            self._computations_data.pop(computation)
            if publish:
                self.logger.info('Unpublish computation %s from agent %s',
                                 computation, agent)
                # if we publish the removal, we must unsubscribe first, so that
                # we don't get a notification from the directory.
                self.unsubscribe_computation(computation, None)

                self.discovery_computation.send_to_directory(
                    UnPublishComputationMessage(computation, agent))
            else:
                self.logger.info('un-register local computation %s from '
                                 'agent %s', computation, agent)

        except KeyError:
            self.logger.info('Attempting to unregister an unknown '
                             'computation %s', computation)

    def subscribe_computation(self, computation: ComputationName,
                              cb: Optional[DiscoveryCallBack]= None,
                              one_shot: bool=False)-> DiscoveryCallBack:
        """
        Subscribe to a computation on the directory.

        When subscribed to an computation, the discovery instance is will be
        notified of any registration or un-registration for this computation.

        The computation may or may not be registered on the directory,
        if it is already registered the discovery instance will be notified
        immediately, if it is not registered yet, this discovery instance will
        be notified (and updated) once the computation is registered (which may
        never happen).

        If a callback is provided, it will be called when receiving such
        notification, otherwise the local discovery will simply be updated.

        The callback must be a callable accepting 3 strings as parameters. The
        first parameter will be 'computation_added' or 'computation_removed'
        and the second and third parameter will be the computation's name and
        the name of the agent hosting it.


        Parameters
        ----------
        computation: str
            The computation name
        cb: DiscoveryCallBack
            An optional callback as a callable accepting 3 strings as
            parameters.
        one_shot: bool
            If true, the callback will be discarded after one call. Notice
            that the discovery instance will still be updated by the
            directory for this computation.

        Returns
        -------
        Callable
            the callback, or None if no callback was given.

        See Also
        --------
        Discovery.unsubscribe_computation
        """
        self.logger.debug('Subscribe to computation %s : %s, %s',
                          computation, cb, one_shot)
        already_subscribed = computation in self._computation_cbs
        if cb is not None:
            self._computation_cbs[computation].append((cb, one_shot))
        if not already_subscribed or cb is None:
            self.discovery_computation.send_to_directory(
                SubscribeComputationMessage(computation, True))
        return cb

    def unsubscribe_computation(self, computation: ComputationName,
                                cb: Optional[DiscoveryCallBack]=None):
        """
        Unsubscribe callbacks for a computation

        Parameters
        ----------
        computation
        cb: callback or None
            If a callback is given, only this callback is removed, otherwise,
            all callback registered for this computation are removed.

        """
        self.logger.debug('Un-subscribe from computation %s : %s',
                          computation, cb)
        removed = 0
        if computation in self._computation_cbs:
            for r_cb, one_shot in self._computation_cbs[computation][:]:
                if cb is None or r_cb == cb:
                    self._computation_cbs[computation].remove((r_cb, one_shot))
                    removed += 1
            if removed:
                if not self._computation_cbs[computation]:
                    self._computation_cbs.pop(computation)
                    self.logger.debug('No callback left for %s, unsubscribe '
                                      'on directory', computation)
                    self.discovery_computation.send_to_directory(
                        SubscribeComputationMessage(computation, False))
            elif cb is not None:
                raise ValueError(
                    'No corresponding callback found for computation %s : %s',
                    computation, cb)
        else:
            self.discovery_computation.send_to_directory(
                SubscribeComputationMessage(computation, False))
        return removed

    def register_replica(self, replica: ComputationName, agent: AgentName,
                         publish=True)\
            -> None:
        """
        Publish replica for `computation` hosted on `agent`

        Parameters
        ----------
        agent
        replica
        publish: bool
            If True, the replica registration will be published on the
            directory.


        """
        if replica not in self._computations_data:
            raise UnknownComputation('Cannot register replica for '
                                     'unknown computation ' + replica)
        agent = self.own_agent if agent is None else agent

        is_change = agent not in self._replicas_data[replica]

        self._replicas_data[replica].add(agent)
        if publish:
            self.logger.info('Publishing replica %s hosted on %s',
                             replica, agent)
            self.discovery_computation.send_to_directory(
                PublishReplicaMessage(replica, agent, True))
        else:
            pass

        # Fire callbacks only if there was an actual change
        if not is_change:
            return
        if replica in self._replicas_cbs:
            for cb, oneshot in self._replicas_cbs[replica]:
                self.logger.debug('fire replica_added callback for %s on %s : '
                                  '%s', replica, agent, cb)
                cb('replica_added', replica, agent)
            # Remove all one-shot callback for this replica
            self._replicas_cbs[replica][:] = \
                [(cb, oneshot)
                 for cb, oneshot in self._replicas_cbs[replica]
                 if not oneshot]

    def unregister_replica(self, replica: ComputationName, agent: AgentName,
                           publish: bool=True):
        """
        Un-registers a replica hosted on `agent`.

        Notes
        -----
        If the computation is unknown not exception is logged (& info is logged)

        Parameters
        ----------
        replica: str
            The name of the computation whose replica is unregistered
        agent: str
            The name of the agent that hosts the replica.
        publish: bool
            If True, the replica un-registration will be published on the
            directory.
        """

        if replica not in self._replicas_data:
            self.logger.info('Attempting to unregister an unknown '
                             'replica %s', replica)
            return
        if agent not in self._replicas_data[replica]:
            self.logger.info('Attempting to unregister replica %s on unknown '
                             'agent %s', replica, agent)
            return

        self._replicas_data[replica].remove(agent)
        # Fire callbacks if any.
        if replica in self._replicas_cbs:
            for cb, _ in self._replicas_cbs[replica]:
                self.logger.debug('fire replica_removed callback '
                                  'for %s on %s, %s ', replica, agent, cb)
                cb('replica_removed', replica, agent)

        if publish:
            self.logger.info('Unpublish replica %s from agent %s',
                             replica, agent)

            self.discovery_computation.send_to_directory(
                PublishReplicaMessage(replica, agent, False))
        else:
            self.logger.info('un-register local replica %s from '
                             'agent %s', replica, agent)

    def subscribe_replica(self, replica: ComputationName,
                          cb: Optional[DiscoveryCallBack]=None,
                          one_shot: bool=False) -> DiscoveryCallBack:
        """
        Subscribe to replicas of computation named `replica`
        Parameters
        ----------
        replica: str
            name of the computation whose we are replicas we
            subscribe to.
        cb: DiscoveryCallBack
            An optional callback as a callable accepting 3 strings as
            parameters.
        one_shot: bool
            If true, the callback will be discarded after one call. Notice
            that the discovery instance will still be updated by the
            directory for this computation's replicas (until
            `unsubscribe_replica` is called).

        Returns
        -------
        Callable
            the callback, or None if no callback was given. This allows
            chaining calls.

        See Also
        --------
        Discovery.unsubscribe_replica
        """
        self.logger.debug('Subscribe to replica for computation %s : %s, %s',
                          replica, cb, one_shot)
        already_subscribed = replica in self._replicas_cbs
        if cb is not None:
            self._replicas_cbs[replica].append((cb, one_shot))
        if not already_subscribed or cb is None:
            self.discovery_computation.send_to_directory(
                SubscribeReplicaMessage(replica, True))
        return cb

    def unsubscribe_replica(self, replica: ComputationName,
                            cb: Optional[DiscoveryCallBack]=None):
        """
        Unsubscribe callbacks for a replicas of a computation

        Parameters
        ----------
        replica: str
            name of the computation whose we are replicas we
            unsubscribe from.
        cb: callback or None
            If a callback is given, only this callback is removed, otherwise,
            all callback registered for this replica are removed.

        """
        self.logger.debug('Un-subscribe from replica %s : %s',
                          replica, cb)
        removed = 0
        if replica in self._replicas_cbs:
            for r_cb, one_shot in self._replicas_cbs[replica][:]:
                if cb is None or r_cb == cb:
                    self._replicas_cbs[replica].remove((r_cb, one_shot))
                    removed += 1
            if removed:
                if not self._replicas_cbs[replica]:
                    self._replicas_cbs.pop(replica)
                    self.logger.debug('No callback left for %s, unsubscribe '
                                      'on directory', replica)
                    self.discovery_computation.send_to_directory(
                        SubscribeComputationMessage(replica, False))
                    # remove all knowledge of current replicas as we are not
                    #  subscribed any more
                    self._replicas_data.pop(replica)
            elif cb is not None:
                raise ValueError(
                    'No corresponding callback found for replica %s : %s',
                    replica, cb)
        else:
            self.discovery_computation.send_to_directory(
                SubscribeReplicaMessage(replica, False))
            # remove all knowledge of current replicas as we are not
            #  subscribed any more
            self._replicas_data.pop(replica)
        return removed

    def replica_agents(self, replica: ComputationName) -> Set[AgentName]:
        """
        List of agents hosting a replica for `computation`

        Parameters
        ----------
        replica

        Returns
        -------

        """
        if replica not in self._computations_data:
            raise UnknownComputation(replica)

        return set(self._replicas_data[replica])
