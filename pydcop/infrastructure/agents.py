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
Base 'Agent' classes.

An Agent instance is a stand-alone autonomous object. It hosts computations,
which send messages to each other.
Each agent has its own thread, which is used to handle messages as they are
dispatched to computations hosted on this agent.



"""

import logging
import sys
import threading
import traceback
import random
from functools import partial
from importlib import import_module
from threading import Thread
from time import perf_counter, sleep
from typing import Dict, List, Optional, Union, Callable, Tuple

from collections import defaultdict

from pydcop.algorithms import AlgorithmDef, ComputationDef, load_algorithm_module
from pydcop.dcop.objects import AgentDef, create_binary_variables
from pydcop.dcop.objects import BinaryVariable
from pydcop.dcop.relations import Constraint
from pydcop.infrastructure.Events import event_bus
from pydcop.infrastructure.communication import Messaging, \
    CommunicationLayer, UnreachableAgent
from pydcop.infrastructure.computations import MessagePassingComputation, \
    build_computation
from pydcop.infrastructure.discovery import Discovery, UnknownComputation, \
    UnknownAgent, _is_technical
from pydcop.infrastructure.ui import UiServer
from pydcop.reparation import create_computation_hosted_constraint, \
    create_agent_capacity_constraint, create_agent_hosting_constraint, \
    create_agent_comp_comm_constraint


class AgentException(Exception):
    pass


class Agent(object):
    """
    Object representing an agent.

    An agent communicates with other agents though messages, using a
    `CommunicationLayer`
    An agent hosts message passing computations and run these computations on
    its own thread.

    Notes
    -----
    An agent does not necessarily need to known it's own definition (see
    agent_def argument) but is needs it for some use like replication in
    resilient DCOP.

    Parameters
    ----------
    name: str
        name of the agent
    comm: CommunicationLayer
        object used to send and receive messages
    agent_def: AgentDef
        definition of this agent, optional
    ui_port: int
        the port on which to run the ui-server. If not given, no ui-server is
        started.
    delay: int
        An optional delay between message delivery, in second. This delay
        only applies to algorithm's messages and is useful when you want to
        observe (for example with the GUI) the behavior of the algorithm at
        runtime.
    daemon: boolean
        indicates if the agent should use a daemon thread (defaults to False)

    See Also
    --------
    MessagePassingComputation, CommunicationLayer

    """
    def __init__(self, name,
                 comm: CommunicationLayer,
                 agent_def: AgentDef=None,
                 ui_port: int=None,
                 delay: float=None,
                 daemon: bool=False):
        self._name = name
        self.agent_def = agent_def
        self.logger = logging.getLogger('pydcop.agent.' + name)
        self.agt_metrics = AgentMetrics()

        # Setup communication and discovery
        self._comm = comm
        self.discovery = Discovery(self._name, self.address)
        self._comm.discovery = self.discovery
        self._messaging = Messaging(name, comm, delay=delay)
        self.discovery.discovery_computation.message_sender = \
            self._messaging.post_msg

        # Ui server
        self._ui_port = ui_port
        self._ui_server = None

        self.t = Thread(target=self._run, name='thread_'+name)
        self.t.daemon = daemon
        self._stopping = threading.Event()
        self._shutdown = threading.Event()
        self._running = False
        # _idle means that we have finished to handle all incoming messages
        self._idle = False

        self._computations = {}  # type: Dict[str, MessagePassingComputation]

        self.t_active = 0
        # time when run the first non-technical computation is run
        self._run_t = None
        # time when starting the agent
        self._start_t = None

        # Tasks that must run periodically as {callable: (period, last_run)}
        self._periodic_cb = {}  # type: Dict[Callable, Tuple[float, float]]

        # List of paused computations, any computation whose name is in this
        # list will not receive any message.
        self.paused_computations = []

    @property
    def communication(self)-> CommunicationLayer:
        """
        The communication used by this agent.

        Returns
        -------
        CommunicationLayer
            The communication used by this agent.
        """
        return self._comm

    def add_computation(self, computation: MessagePassingComputation,
                        comp_name=None, publish=True):
        """
        Add a computation to the agent.

        The computation will run on this agent thread and receives messages
        through his Messaging and CommunicationLayer.

        Parameters
        ----------
        computation: a MessagePassingComputation
            the computation to be added

        comp_name: str
            an optional name for the computation, if not given
            computation.name will be used.
        publish: bool
            True (default) is the computation must be published on the
            discovery service.

        """
        comp_name = computation.name if comp_name is None else comp_name
        self.logger.debug('Add computation %s - %s ',
                          comp_name, self._messaging)
        computation.message_sender = self._messaging.post_msg
        computation.periodic_action_handler = self
        self._computations[comp_name] = computation
        self.discovery.register_computation(comp_name, self.name,self.address,
                                            publish=publish)

        # start lookup for agent hosting a neighbor computation
        if hasattr(computation, 'computation_def') and \
                computation.computation_def is not None:
            for n in computation.computation_def.node.neighbors:
                self.discovery.subscribe_computation(n)

        if hasattr(computation, '_on_value_selection'):
            computation._on_value_selection = notify_wrap(
                computation._on_value_selection,
                partial(self._on_computation_value_changed, computation.name))
        if hasattr(computation, '_on_new_cycle'):
            computation._on_new_cycle = notify_wrap(
                computation._on_new_cycle,
                partial(self._on_computation_new_cycle, computation.name))

        computation.finished = notify_wrap(
            computation.finished,
            partial(self._on_computation_finished, computation.name))

        event_bus.send("agents.add_computation."+self.name,
                       (self.name, computation))

    def remove_computation(self, computation: str) -> None:
        """
        Removes a computation from the agent.

        Parameters
        ----------
        computation: str
            the name of the computation

        Raises
        ------
        UnknownComputation
            If there is no computation with this name on this agent

        """
        try:
            comp = self._computations.pop(computation)
        except KeyError:
            self.logger.error(
                'Removing unknown computation %s - current commutations : %s',
                computation, self._computations)
            raise UnknownComputation(computation)
        if comp.is_running:
            comp.stop()
        self.logger.debug('Removing computation %s', comp)
        self.discovery.unregister_computation(computation, self.name)

        event_bus.send("agents.rem_computation."+self.name,
                       (self.name, computation))

    def computations(self, include_technical=False)-> \
            List[MessagePassingComputation]:
        """
        Computations hosted on this agent.

        Parameters
        ----------
        include_technical: bool
            If True, technical computations (like discovery, etc.) are
            included in the list.

        Returns
        -------
        List[MessagePassingComputation]
            A list of computations hosted on this agents. This list is a copy
            and can be safely modified.

        """
        if include_technical:
            return list(self._computations.values())
        else:
            return [c for c in self._computations.values()
                    if not c.name.startswith('_')]

    def computation(self, name: str) -> MessagePassingComputation:
        """
        Get a computation hosted by this agent.

        Parameters
        ----------
        name: str
            The name of the computation.

        Returns
        -------
            The Messaging passing corresponding to the given name.

        Raises
        ------
        UnknownComputation
            if the agent has no computation with this name.


        See Also
        --------
        add_computation
        """
        try:
            return self._computations[name]
        except KeyError:
            self.logger.error('unknown computation %s', name)
            raise UnknownComputation('unknown computation ' + name)

    @property
    def address(self):
        """
        The address this agent can be reached at.

        The type of the address depends on the instance and type of the
        CommunicationLayer used by this agent.

        Returns
        -------
            The address this agent can be reached at.
        """
        return self._comm.address

    def start(self, run_computations = False):
        """
        Starts the agent.

        One started, an agent will dispatch any received message to the
        corresponding target computation.

        Notes
        -----
        Each agent has it's own thread, this will start the agent's thread,
        run the _on_start callback and waits for message. Incoming message are
        added to a queue and handled by calling the _handle_message callback.

        The agent (and its thread) will stop  once stop() has been called and
        he has finished handling the current message, if any.

        See Also
        --------
        _on_start(), stop()

        """
        if self.is_running:
            raise AgentException('Cannot start agent {}, already running '
                                 .format(self.name))
        self.logger.info('Starting agent %s ', self.name)
        self._running = True
        self.run_computations = run_computations
        self._start_t = perf_counter()
        self.t.start()

    def run(self, computations: Optional[Union[str, List[str]]]=None):
        """
        Run computations hosted on this agent.

        Notes
        -----
        Attempting to start an already running computation is harmless : it
        will be logged but will not raise an exception.
        The first time this method is called, timestamp is stored, which is used
        as a reference when computing metrics.

        Parameters
        ----------
        computations: Optional[Union[str, List[str]]]
            An optional computation name or list of computation names. If None,
            all computations hosted on this agent are started.

        Raises
        ------
        AgentException
            If the agent was not started (using agt.start()) before calling
            run().
        UnknownComputation
            If some of the computations are not hosted on this agent. All
            computations really hosted on the agent are started before raising
            this Exception.
        """
        if not self.is_running:
            raise AgentException('Cannot start computation on agent %s which '
                                 'is not started', self.name)

        if computations is None:
            self.logger.info('Starting all computations')
        else:
            if isinstance(computations, str):
                computations = [computations]
            else:
                # avoid modifying caller's variable
                computations = computations[:]
            self.logger.info('Starting computations %s', computations)

        if self._run_t is None:
            # We start counter time only when the first computation is run,
            # to avoid counting idle time when we wait for orders.
            self._run_t = perf_counter()

        on_start_t = perf_counter()
        for c in list(self._computations.values()):
            if computations is None:
                if c.is_running:
                    self.logger.debug(f'Do not start computation {c.name}, already '
                                      'running')
                else:
                    c.start()
            elif c.name in computations:
                if c.is_running:
                    self.logger.debug(f'Do not start computation {c.name}, already '
                                      'running')
                else:
                    c.start()
                computations.remove(c.name)
        # add the time spent in on_start to the active time of the agent.
        self.t_active += perf_counter() - on_start_t

        if computations:
            raise UnknownComputation('Could not start unknown computation %s',
                                     computations)

    @property
    def start_time(self)-> float:
        """
        float:
            timestamp for the first run computation call. This timestamp is
            used as a reference when computing various time-related metrics.
        """
        return self._run_t

    def clean_shutdown(self):
        """
        Perform a clean shutdown of the agent.

        All pending messages are handled before stopping the agent thread.

        This method returns immediately, use `join` to wait until the agent's
        thread has stopped.

        """
        self.logger.debug('Clean shutdown requested')
        self._shutdown.set()
        self._messaging.shutdown()

    def stop(self):
        """
        Stops the agent

        A computation cannot be interrupted while it handle a message,
        as a consequence the agent (and its thread) will stop once it he has
        finished handling the current message, if any.
        """
        self.logger.debug('Stop requested on %s', self.name)
        self._stopping.set()

    def pause_computations(self, computations: Union[str, Optional[List[str]]]):
        """
        Pauses computations.

        Parameters
        ----------
        computations:  Union[str, Optional[List[str]]]
            The name of the computation to pause, or a list of computations
            names. If None, all hosted computation will be paused.

        Raises
        ------
        AgentException
            If the agent was not started (using agt.start()) before calling
            pause_computations().
        UnknownComputation
            If some of the computations are not hosted on this agent. All
            computations really hosted on the agent are paused before raising
            this exception.

        """
        if not self.is_running:
            raise AgentException('Cannot pause computations on agent %s which '
                                 'is not started')

        if computations is None:
            self.logger.info('Pausing all computations')
        else:
            if isinstance(computations, str):
                computations = [computations]
            else:
                computations = computations[:]
            self.logger.info('Pausing computations %s', computations)

        for c in self._computations.values():
            if computations is None:
                if c.is_paused:
                    self.logger.warning('Cannot pause computation %s, already '
                                        'paused', c.name)
                else:
                    c.pause(True)
            elif c.name in computations:
                if c.is_paused:
                    self.logger.warning('Cannot pause computation %s, already '
                                        'paused', c.name)
                else:
                    c.pause(True)
                computations.remove(c.name)

        if computations:
            raise UnknownComputation('Could not pause unknown computation %s',
                                     computations)

    def unpause_computations(self,
                             computations: Union[str, Optional[List[str]]]):
        """
        Un-pause (i.e. resume) computations

        Parameters
        ----------
        computations: Optional[List[str]]
            TThe name of the computation to resume, or a list of computations
            names. If None, all hosted computations will be resumed.

        Raises
        ------
        AgentException
            If the agent was not started (using agt.start()) before calling
            unpause_computations().
        UnknownComputation
            If some of the computations are not hosted on this agent. All
            computations really hosted on the agent are resumed before raising
            this exception.

        """
        if not self.is_running:
            raise AgentException('Cannot resume computations on agent %s which '
                                 'is not started')

        if computations is None:
            self.logger.info('Resuming all computations')
        else:
            if isinstance(computations, str):
                computations = [computations]
            else:
                computations = computations[:]
            self.logger.info('Resuming computations %s', computations)

        for c in self._computations.values():
            if computations is None:
                if not c.is_paused:
                    self.logger.warning('Do not resume computation %s, not '
                                      'paused', c.name)
                else:
                    c.pause(False)
            elif c.name in computations:
                if not c.is_paused:
                    self.logger.warning('Do not resume computation %s, not '
                                      'paused', c.name)
                else:
                    c.pause(False)
                computations.remove(c.name)

        if computations:
            raise UnknownComputation('Could not resume unknown computation %s',
                                     computations)

    @property
    def name(self):
        """
        str:
            The name of the agent.
        """
        return self._name

    @property
    def is_stopping(self)-> bool:
        """
        bool:
            True if the agent is currently stopping (i.e. handling its last
            message).
        """
        return self._stopping.is_set()

    @property
    def is_running(self):
        """
        bool:
            True if the agent is currently running.
        """
        return self._running

    def join(self):
        self.t.join()

    def _on_start(self):
        """
        This method is called when the agent starts.


        Notes
        -----
        This method is meant to be overwritten in subclasses that might need to
        perform some operations on startup. Do NOT forget to call
        `super()._on_start()` ! When `super()._on_start()` return `False`,
        you must also return `False` !

        This method is always run in the agent's thread, even though the
        `start()` method is called from an other thread.

        Returns
        -------
        status: boolean
            True if all went well, False otherwise
        """
        self.logger.debug('on_start for {}'.format(self.name))

        if self._ui_port:
            event_bus.enabled = True
            self._ui_server = UiServer(self, self._ui_port)
            self.add_computation(self._ui_server, publish=False)
            self._ui_server.start()
        else:
            self.logger.debug('No ui server for %s', self.name)

        self._computations[self.discovery.discovery_computation.name] = \
            self.discovery.discovery_computation
        while True:
            # Check _stopping: do not prevent agent form stopping !
            if self._stopping.is_set():
                return False
            try:
                self.discovery.register_computation(
                    self.discovery.discovery_computation.name,
                    self.name, self.address)
            except UnreachableAgent:
                self.logger.warning("Could not reach directory, will retry "
                                    "later")
                sleep(1)
            else:
                break
        self.discovery.register_agent(self.name, self.address)
        self.discovery.discovery_computation.start()

        return True

    def _on_stop(self):
        """
        This method is called when the agent has stopped.

        It is meant to be overwritten in subclasses that might need to
        perform some operations on stop, however, when overwriting it,
        you MUST call `super()._on_stop()`.

        Notes
        -----
        This method always run in the agent's thread. Messages can still be
        sent in this message, but no new message will be received (as the
        agent's thread has stopped)

        """
        self.logger.debug('on_stop for %s with computations %s ',
                          self.name, self.computations())

        # Unregister computations and agent from discovery.
        # This will also unregister any discovery callbacks this agent may still
        # have.
        for comp in self.computations():
            comp.stop()
            if not _is_technical(comp.name):
                try:
                    self.discovery.unregister_computation(comp.name)
                except UnreachableAgent:
                    # when stopping the agent, the orchestrator / directory might have
                    # already left.
                    pass

        if self._ui_server:
            self._ui_server.stop()

        try:
            # Wait a bit to make sure that the stopped message can reach the
            # orchestrator before unregistration.
            sleep(0.5)
            self.discovery.unregister_agent(self.name)
        except UnreachableAgent:
            # when stopping the agent, the orchestrator / directory might have
            # already left.
            pass

    def _on_computation_value_changed(self, computation: str, value,
                                      cost, cycle):
        """Called when a computation selects a new value """
        pass

    def _on_computation_new_cycle(self, computation, *args, **kwargs):
        """Called when a computation starts a new cycle"""
        pass

    def _on_computation_finished(self, computation: str,
                                 *args, **kwargs):
        """
        Called when a computation finishes.

        This method is meant to be overwritten in sub-classes.

        Parameters
        ----------
        computation: str
            name of the computation that just ended.
        """
        pass

    def _handle_message(self, sender_name: str, dest_name: str, msg, t):
        # messages are delivered even to computations which have reached their
        # stop condition. It's up the the algorithm to decide if it wants to
        # handle the message.

        dest = self.computation(dest_name)
        dest.on_message(sender_name, msg, t)

    def metrics(self):
        if self._run_t is None:
            activity_ratio = 0
        else:
            total_t = perf_counter() - self._run_t
            activity_ratio = self.t_active / (total_t)
        own_computations = { c.name for c in self.computations(include_technical=True)}
        m = {
            'count_ext_msg': {k: v
                              for k, v in self._messaging.count_ext_msg.items()
                              if k in own_computations},
            'size_ext_msg': {k: v
                             for k, v in self._messaging.size_ext_msg.items()
                             if k in own_computations},
            # 'last_msg_time': self._messaging.last_msg_time,
            'activity_ratio': activity_ratio,
            'cycles': {c.name: c.cycle_count for c in self.computations()}
        }
        return m

    def messages_count(self, computation: str):
        return self._messaging.count_ext_msg[computation]

    def messages_size(self, computation: str):
        return self._messaging.size_ext_msg[computation]

    def set_periodic_action(self, period: float, cb: Callable):
        """
        Set a periodic action.

        The callback `cb` will be called every `period` seconds. The delay
        is not strict. The handling of a message is never interrupted,
        if it takes longer than `period`, the callback will be delayed and
        will only be called once the task has finished.

        Parameters
        ----------
        period: float
            a period in second
        cb: Callable
            a callback with no argument

        Returns
        -------
        handle:
            An handle that can be used to remove the periodic action.
            This handle is actually the callback object itself.

        """
        assert period != None
        assert cb != None
        self.logger.debug("Add periodic action %s - %s ", period, cb)
        self._periodic_cb[cb] = (period, perf_counter())
        return cb

    def remove_periodic_action(self, handle):
        """
        Remove a periodic action

        Parameters
        ----------
        handle:
            the handle returned by set_periodic_action

        """
        self.logger.debug("Remove action %s ", handle)
        self._periodic_cb.pop(handle)

    def _run(self):
        self.logger.debug('Running agent ' + self._name)
        full_msg = None
        try:
            self._running = True
            self._on_start()
            if self.run_computations:
                self.run()
            while not self._stopping.is_set():
                # Process messages, if any
                full_msg, t = self._messaging.next_msg(0.05)
                if full_msg is None:
                    self._idle = True
                    if self._shutdown.is_set():
                        self.logger.info("No message during shutdown, "
                                         "stopping agent thread")
                        break
                else:

                    current_t = perf_counter()
                    try:
                        sender, dest, msg, _ = full_msg
                        self._idle = False
                        if not self._stopping.is_set():
                            self._handle_message(sender, dest, msg, t)
                    finally:
                        if self._run_t is not None:
                            e = perf_counter()
                            msg_duration = e - current_t
                            self.t_active += msg_duration
                            if msg_duration > 1:
                                self.logger.warning(
                                    'Long message handling (%s) : %s',
                                    msg_duration, msg)

                self._process_periodic_action()

        except Exception as e:
            self.logger.error('Thread %s exits With error : %s \n '
                              'Was handling message %s ',
                              self.name, e, full_msg)
            self.logger.error(traceback.format_exc())
            if hasattr(self, 'on_fatal_error'):
                self.on_fatal_error(e)

        except:  # catch *all* exceptions
            e = sys.exc_info()[0]
            self.logger.error('Thread exits With un-managed error : %s', e)
            self.logger.error(e)
        finally:
            self._running = False
            self._comm.shutdown()
            self._on_stop()
            self.logger.info('Thread of agent %s stopped', self._name)

    def _process_periodic_action(self):
        # Process periodic action. Only once the agents runs the
        # computations (i.e. self._run_t is not None)
        ct = perf_counter()
        if self._start_t is not None :
            for cb, (p, last_t) in list(self._periodic_cb.items()):
                if ct - last_t >= p:
                    # self.logger.debug('periodic cb %s, %s %s ', cb, ct, last_t)
                    # Must update the cb entry BEFORE calling the cb, in case
                    # the cb attemps to modify (e.g. remove) it's own entry by
                    # calling remove_periodic_action
                    self._periodic_cb[cb] = (p, ct)
                    cb()

    def is_idle(self):
        """
        Indicate if the agent is idle. An idle agent is an agent which has no
        pending messages to handle.

        :return: True if the agent is idle, False otherwise
        """
        return self._idle

    def __str__(self):
        return 'Agent: '+self._name

    def __repr__(self):
        return 'Agent: ' + self._name


def notify_wrap(f, cb):

    def wrapped(*args, **kwargs):
        f(*args, **kwargs)
        cb(*args, **kwargs)
    return wrapped


class AgentMetrics(object):
    """
    AgentMetrics listen to events from the event_bus to consolidate metrics.

    """

    def __init__(self):
        self._computation_msg_rcv = defaultdict(lambda : (0,0))
        self._computation_msg_snd = defaultdict(lambda : (0,0))

        event_bus.subscribe('computations.message_rcv.*',
                            self._on_computation_msg_rcv)
        event_bus.subscribe('computations.message_snd.*',
                            self._on_computation_msg_snd)


    def computation_msg_rcv(self, computation: str):
        return self._computation_msg_rcv[computation]

    def computation_msg_snd(self, computation: str):
        return self._computation_msg_snd[computation]

    def _on_computation_msg_rcv(self, topic: str, msg_event):
        computation, msg_size = msg_event
        prev_count , prev_size = self._computation_msg_rcv[computation]
        self._computation_msg_rcv[computation] = \
            prev_count+1, prev_size+ msg_size

    def _on_computation_msg_snd(self, topic: str, msg_event):
        computation, msg_size = msg_event
        prev_count , prev_size = self._computation_msg_snd[computation]
        self._computation_msg_snd[computation] = \
            prev_count+1, prev_size+ msg_size


from pydcop.computations_graph import constraints_hypergraph as chg

repair_algo = load_algorithm_module('mgm2')


class RepairComputationRegistration(object):

    def __init__(self, computation: MessagePassingComputation,
                 status: str, candidate: str):
        self.computation = computation
        self.status = status
        self.candidate = candidate


class ResilientAgent(Agent):
    """

    An agent that supports resiliency by replicating it's computations.

    Parameters
    ----------
    name: str
        name of the agent
    comm: CommunicationLayer
        object used to send and receive messages
    agent_def: AgentDef
        definition of this agent, optional
    ui_port: int
        the port on which to run the ui-server. If not given, no ui-server is
        started.
    replication: str
        name of the replication algorithm
    delay: int
        An optional delay between message delivery, in second. This delay
        only applies to algorithm's messages and is useful when you want to
        observe (for example with the GUI) the behavior of the algorithm at
        runtime.

    """

    def __init__(self, name: str, comm: CommunicationLayer,
                 agent_def: AgentDef, replication: str, ui_port=None,
                 delay: float=None):
        super().__init__(name, comm, agent_def, ui_port=ui_port, delay=delay)
        self.replication_comp = None
        if replication is not None:
            self.logger.debug('deploying replication computation %s',
                              replication)
            # DCOP computations will be added to the replication computation
            # as they are deployed.
            algo_module = import_module('pydcop.replication.{}'
                                        .format(replication))
            self.replication_comp = algo_module.build_replication_computation(
                self, self.discovery)

            # self.add_computation(self.replication_comp)
            # Do not start the computation yet, the agent is not event started

            self._repair_computations =\
                {}  # type: Dict[str, RepairComputationRegistration]
            # the replication level will be set by the when requested to
            # replicate, by the ReplicateComputationsMessage
            self._replication_level = None

            # Register notification for when all computations have been
            # replicated.
            self.replication_comp.replication_done = notify_wrap(
                self.replication_comp.replication_done,
                self._on_replication_done)

    def _on_start(self):
        """
        See Also
        --------
        Agent._on_start

        Returns
        -------
        status

        """
        self.logger.debug('Resilient agent on_start')
        if not super()._on_start():
            return False
        if self.replication_comp is not None:
            self.add_computation(self.replication_comp)
            self.replication_comp.start()
        return True

    def _on_stop(self):
        if self.replication_comp is not None:
            self.replication_comp.stop()
            self.discovery.unregister_computation(self.replication_comp.name)
        super()._on_stop()

    def add_computation(self, computation: MessagePassingComputation,
                        comp_name=None, publish=True):
        """
        Add a computation to the agent.

        See Also
        --------
        Agent.add_computation

        Parameters
        ----------
        computation
        comp_name
        publish

        Returns
        -------

        """
        super().add_computation(computation, comp_name, publish)
        if self.replication_comp is not None \
                and not computation.name.startswith('_')\
                and not computation.name.startswith('B'):
            # FIXME : find a better way to filter out repair computation than
            # looking at the first character (B).
            self.replication_comp.add_computation(computation.computation_def,
                                                  computation.footprint())

    def remove_computation(self, computation: str):
        if self.replication_comp is not None \
                and not computation.startswith('_'):
            self.replication_comp.remove_computation(computation)
        super().remove_computation(computation)

    def replicate(self, k: int):
        if self.replication_comp is not None:
            self._replication_level = k
            self.replication_comp.replicate(k)

    def setup_repair(self, repair_info):
        self.logger.info('Setup repair %s', repair_info)
        # create computation for the reparation dcop
        # The reparation dcop uses a dcop algorithm where computations maps to
        # variable (in order to have another dcop distribution problem) and use
        # binary variable for each candidate computation.
        # This agent will host one variable-computation for each
        # binary variable x_i^m indicating if the candidate computation x_i
        # is hosted on this agent a_m. Notice that by construction,
        # the agent already have a replica for all the candidates x_i.

        # The reparation dcop includes several constraints and variables:
        # Variables
        #  * one binary variable for each orphaned computation
        # Constraints
        #  * hosted constraints : one for each candidate computation
        #  * capacity constraint : one for this agent
        #  * hosting costs constraint : one for this agent
        #  * communication constraint
        #
        # For reparation, we use a dcop algorithm where computations maps to
        # variables of the dcop. On this agent, we host the computations
        # corresponding to the variables representing the orphaned computation
        # that could be hosted on this agent (aka candidate computation).
        # Here, we use MGM

        own_name = self.name

        # `orphaned_binvars` is a map that contains binary variables for
        # orphaned computations.
        # Notice that it only contains variables for computations
        # that this agents knows of, i.e. computations that could be hosted
        # here (aka candidate computations) or that depends on computations
        # that could be hosted here.
        # There is one binary variable x_i^m for each pair (x_i, a_m),
        # where x_i is an orphaned computation and a_m is an agent that could
        #  host x_i (i.e. has a replica of x_i).
        orphaned_binvars = {}  # type: Dict[Tuple, BinaryVariable]

        # One binary variable x_i^m for each candidate computation x_i that
        # could be hosted on this agent a_m. Computation for these variables
        # will be hosted in this agent. This is a subset of orphaned_binvars.
        candidate_binvars = {}  # type: Dict[Tuple, BinaryVariable]

        # Agent  that will host the computation for each binary var.
        # it is a dict { bin var name : agent_name }
        # agt_hosting_binvar = {}  # type: Dict[str, str]

        # `hosted_cs` contains hard constraints ensuring that all candidate
        # computations are hosted:
        hosted_cs = {}  # type: Dict[str, Constraint]
        for candidate_comp, candidate_info in repair_info.items():

            try:
                # This computation is not hosted any more, if we had it in
                # discovery, forget about it but do not publish this
                # information, this agent is not responsible for updatings
                # other's discovery services.
                self.discovery.unregister_computation(candidate_comp,
                                                      publish=False)
            except UnknownComputation:
                pass

            agts, _, neighbors = candidate_info
            # One binary variable for each candidate agent for computation
            # candidate_comp:
            v_binvar = create_binary_variables(
                'B', ([candidate_comp], candidate_info[0]))
            # Set initial values for binary decision variable
            for v in v_binvar.values():
                v._intial_value = 1 if random.random() < 1/3 else 0


            orphaned_binvars.update(v_binvar)

            # the variable representing if the computation will be hosted on
            # this agent:
            candidate_binvars[(candidate_comp, own_name)] = \
                v_binvar[(candidate_comp, own_name)]

            # the 'hosted' hard constraint for this candidate variable:
            hosted_cs[candidate_comp] =\
                create_computation_hosted_constraint(candidate_comp, v_binvar)
            self.logger.debug('Hosted hard constraint for computation %s : %r',
                              candidate_comp, hosted_cs[candidate_comp])

            # One binary variable for each pair (x_j, a_n) where x_j is an
            # orphaned neighbors of candidate_comp and a_n is an agent that
            # could host a_n:
            for neighbor in neighbors:
                v_binvar = create_binary_variables(
                    'B', ([neighbor], neighbors[neighbor]))
                orphaned_binvars.update(v_binvar)

        self.logger.debug('Binary variable for reparation %s ',
                          orphaned_binvars)
        # Agent  that will host the computation for each binary var.
        # it is a dict { bin var name : agent_name }
        agt_hosting_binvar = {v.name: a
                              for (_, a), v in orphaned_binvars.items()}
        self.logger.debug('Agents hosting the computations for these binary '
                          'variables : %s ', agt_hosting_binvar)

        # The capacity (hard) constraint for this agent. This ensures that the
        # capacity of the current agent will not be overflown by hosting too
        # many candidate computations. This constraints depends on the binary
        # variables for the candidate computations.
        remaining_capacity = self.agent_def.capacity - \
            sum(c.footprint() for c in self.computations())
        self.logger.debug('Remaining capacity on agent %s : %s',
                          self.name, remaining_capacity)

        def footprint_func(c_name: str):
            # We have a replica for these computation, we known its footprint.
            return self.replication_comp.hosted_replicas[c_name][1]

        capacity_c = create_agent_capacity_constraint(
            own_name, remaining_capacity, footprint_func,
            candidate_binvars)
        self.logger.debug('Capacity constraint for agt %s : %r',
                          self.name, capacity_c)

        # Hosting costs constraint for this agent. This soft constraint is
        # used to minimize the hosting costs on this agent ; it depends on
        # the binary variables for the candidate computations.
        hosting_c = create_agent_hosting_constraint(
            own_name, self.agent_def.hosting_cost,
            candidate_binvars)
        self.logger.debug('Hosting cost constraint for agt %s : %r',
                          self.name, hosting_c)

        # The communication constraint. This soft constraints is used to
        # minimize the communication cost on this agent. As communication
        # cost depends on where computation on both side of an edge are
        # hosted, it also depends on the binary variables for orphaned
        # computations that could not be hosted here.
        def comm_func(candidate_comp: str, neighbor_comp: str, agt: str):
            # returns the communication cost between the computation
            # candidate_name hosted on the current agent and it's neighbor
            # computation neigh_comp hosted on agt.
            route_cost = self.agent_def.route(agt)

            comp_def = self.replication_comp.replicas[candidate_comp]
            algo = comp_def.algo.algo
            algo_module = load_algorithm_module(algo)
            communication_load = algo_module.communication_load

            msg_load = 0
            for l in comp_def.node.neighbors:
                if l == neighbor_comp:
                    msg_load += communication_load(comp_def.node, neighbor_comp)

            com_load = msg_load * route_cost

            return com_load

        # Now that we have the variables and constraints, we can create
        # computation instances for each of the variable this agent is
        # responsible for, i.e. the binary variables x_i^m that correspond to
        # the candidate variable x_i (and a_m is the current agent)
        self._repair_computations.clear()
        algo_def = AlgorithmDef.build_with_default_param(
            repair_algo.algorithm_name,
            {'stop_cycle': 20, 'threshold': 0.2},
            mode='min',
            parameters_definitions=repair_algo.algo_params)
        for (comp, agt), candidate_var in candidate_binvars.items():
            self.logger.debug('Building computation for binary variable %s ('
                              'variable %s on %s)', candidate_var, comp, agt)
            comm_c = create_agent_comp_comm_constraint(
                agt, comp, repair_info[comp], comm_func, orphaned_binvars)
            self.logger.debug('Communication constraint for computation %s '
                              'on agt %s : %r', comp, self.name, comm_c)
            constraints = [comm_c, hosting_c, capacity_c, hosted_cs[comp]]
            # constraints.extend(hosted_cs.values())
            self.logger.debug('Got %s Constraints for var %s :  %s ',
                              len(constraints), candidate_var, constraints)

            node = chg.VariableComputationNode(candidate_var, constraints)
            comp_def = ComputationDef(node, algo_def)
            computation = repair_algo.build_computation(comp_def)
            self.logger.debug('Computation for %s : %r ',
                          candidate_var, computation)

            # add the computation on this agents and register the neighbors
            self.add_computation(computation, publish=True)
            self._repair_computations[computation.name] = \
                RepairComputationRegistration(computation, 'ready', comp)
            for neighbor_comp in node.neighbors:
                neighbor_agt = agt_hosting_binvar[neighbor_comp]
                try:
                    self.discovery.register_computation(
                        neighbor_comp, neighbor_agt,
                        publish=False)
                except UnknownAgent:
                    # If we don't know this agent yet, we must perform a lookup
                    # and only register the computation once found.
                    # Note the use of partial, to force the capture of
                    # neighbor_comp.
                    def _agt_lookup_done(comp, evt, evt_agt, _):
                        if evt == 'agent_added':
                            self.discovery.register_computation(
                                comp, evt_agt, publish=False)
                    self.discovery.subscribe_agent(
                        neighbor_agt,
                        partial(_agt_lookup_done, neighbor_comp),
                        one_shot=True)

        self.logger.info('Repair setup done one %s, %s computations created, '
                         'inform orchestrator', self.name,
                         len(candidate_binvars))
        return candidate_binvars

    def repair_run(self):
        self.logger.info('Agent runs Repair dcop computations')
        comps = list(self._repair_computations.values())
        for c in comps:
            c.computation.start()
            c.status = 'started'

    def _on_replication_done(self, replica_hosts: Dict[str, List[str]]):
        """
        Called when all computations have been replicated.

        This method method is meant to the overwritten in subclasses.

        Parameters
        ----------

        replica_hosts: a map { computation name -> List of agt name }
            For each active computation hosted by this agent, this map
            contains a list of agents that have been selected to host a
            replica.
        """
        self.logger.info('Replica distribution finished for agent '
                         '%s  : %s (level requested : %s)', self.name,
                         replica_hosts, self._replication_level)
        rep_levels = {computation: len(replica_hosts[computation])
                      for computation in replica_hosts}
        if not all([level >= self._replication_level
                    for level in rep_levels.values()]):
            self.logger.warning('Insufficient replication for computations: '
                                '%s ',
                               rep_levels)

    def _on_computation_finished(self, computation: str,
                                 *args, **kwargs):
        self.logger.debug('Computation %s has finished', computation)

        if self.replication_comp and computation in self._repair_computations:
            self._on_repair_computation_finished(computation)

    def _on_repair_computation_finished(self, computation: str):
        repair_comp = self._repair_computations[computation]
        repair_comp.status = 'finished'

        # deploy the computation if it was selected during reparation:
        if repair_comp.computation.current_value == 1:
            self.logger.info('Reparation: computation %s selected on %s',
                             repair_comp.candidate, self.name)
            comp_def = self.replication_comp.replicas[repair_comp.candidate]
            self.logger.info('Deploying computation %s locally with '
                             'definition , %r', repair_comp.candidate,
                             comp_def)
            comp = build_computation(comp_def)
            self.add_computation(comp, publish=True)
        else:
            self.logger.info('Reparation: computation %s NOT selected on '
                             '%s', repair_comp.candidate, self.name)
        # Remove replica: it will be re-replicated by its new host.
        self.replication_comp.remove_replica(repair_comp.candidate)

        if all(c.status == 'finished'
               for c in self._repair_computations.values()):

            selected_computations = \
                [c.candidate for c in self._repair_computations.values()
                 if c.computation.current_value == 1]
            self.logger.info('All repair computations have finished, '
                             'selected computation : %s',
                             selected_computations)

            metrics = self.metrics()
            print(f" metrics repair {self.name} - {metrics}")
            repair_metrics = {'count_ext_msg' : {}, 'size_ext_msg': {} , 'cycles' :{}}

            for c in self._repair_computations.values():
                c_name = c.computation.name
                if c_name in metrics['count_ext_msg']:
                    repair_metrics['count_ext_msg'][c_name] = metrics['count_ext_msg'][c_name]
                else:
                    repair_metrics['count_ext_msg'][c_name] = 0
                if c_name in metrics['size_ext_msg']:
                    repair_metrics['size_ext_msg'][c_name] = metrics['size_ext_msg'][c_name]
                else:
                    repair_metrics['size_ext_msg'][c_name] = 0
                if c_name in metrics['cycles']:
                    repair_metrics['cycles'][c_name] = metrics['cycles'][c_name]
                else:
                    repair_metrics['cycles'][c_name] = 0

            print(f" {self.name} : metrics after repair  {repair_metrics}")
            self._on_repair_done(selected_computations, repair_metrics)

            if selected_computations:
                self.logger.info('Re-replicate newly activated computations '
                                 'on  %s : %s , level %s', self.name,
                                 selected_computations,
                                 self._replication_level)
                try:
                    self.replication_comp.replicate(self._replication_level,
                                                    selected_computations)
                except UnknownComputation:
                    # avoid crashing if one of the neighbor comp is not repaired yet
                    pass
                self.logger.info('Starting newly activated computations on '
                                 '%s : %s ', self.name,
                                 selected_computations)
                for selected in selected_computations:
                    self.computation(selected).start()
                    self.computation(selected).pause()

            # Remove / undeploy repair comp once repaired
            for repair_comp in self._repair_computations.values():
                self.remove_computation(repair_comp.computation.name)
            self._repair_computations.clear()

    def _on_repair_done(self, selected_computations: List[str]):
        """
        Called when all repair computations have finished.

        This method method is meant to the overwritten in subclasses.

        """
        pass


class RepairComputation(MessagePassingComputation):
    """

    """

    def __init__(self, agent: ResilientAgent):
        super().__init__('_resilience_' + self.agent.name)
        self.agent = agent
        self.logger = logging.getLogger('pydcop.agent.repair.'+agent.name)
        self._handlers = {
            #'replication': self._on_replication,
            # 'setup_repair': self._on_setup_repair,
            # 'repair_run': self._on_repair_run,
        }

    @property
    def type(self):
        return 'replication'

    def on_message(self, var_name, msg, t):
        self._handlers[msg.type](msg)

    def footprint(self):
        return 0

    def replication_done(self, replica_hosts: Dict[str, List[str]]):
        """
        Called when all computations have been replicated.

        The replication algorithm only selects agents to host replicas,
        here we send the actual computations definitions to the agents
        selected to host a replica.

        We also send the obtained replication to the orchestrator.

        Parameters
        ----------

        replica_hosts: a map { computation name -> List of agt name }
            For each active computation hosted by this agent, this map
            contains a list of agents that have been selected to host a
            replica.
        """
        self.logger.info('Replica distribution finished for agent '
                         '%s  : %s', self.name, replica_hosts)
        # self.agent.on_replication_done()
        # dist_msg = ComputationReplicatedMessage(self.name, replica_hosts)
        # self.message_sender.post_send_to_orchestrator(dist_msg)
