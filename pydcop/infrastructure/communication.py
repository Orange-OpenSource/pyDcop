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


import json
import logging
import socket
from collections import namedtuple, defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from json import JSONDecodeError
from queue import Empty, PriorityQueue
from threading import Thread
from time import perf_counter, sleep
from typing import Tuple, Dict, Optional

import requests
from requests.exceptions import ConnectionError

from pydcop.infrastructure.discovery import UnknownComputation, UnknownAgent
from pydcop.utils.simple_repr import simple_repr, from_repr

logger = logging.getLogger("infrastructure.communication")

ComputationMessage = namedtuple(
    "ComputationMessage", ["src_comp", "dest_comp", "msg", "msg_type"]
)


class CommunicationLayer(object):
    """
    Base class for CommunicationLayer objects.

    CommunicationLayer objects are used to sent messages from one agent to
    another. Each agent should have it's own CommunicationLayer instance.

    The behavior on message sending failure can be specified when building
    the instance and overridden for a specific message when calling
    `send_msg`, with the `òn_error` parameter, which accept the
    following values:

    * 'ignore': `send_msg` always return True even if the message could not be
      delivered
    * 'fail': if the target agent could not be found, raise a `UnknownAgent`
      exception. If the target agent does not host the computation for the
      message (and he answered the request with the corresponding error
      code), raise an `UnknownComputation` exception.
    * 'retry': `send_msg` returns `True` only if the message was delivered
      successfully and the target agent host the computation. Otherwise the
      message is kept and the `CommunicationLayer` will try to send it
      latter. This new attempt will be done when calling `register`
      or `retry` for the target agent.

    Notes
    -----

    To be usable a Communication Layer needs a Discovery instance in order to
    have access to other agents address. This instance is not given directly
    in the constructor as it depend on the agent that will be using the
    CommunicationLayer instance, even though the CommunicationLayer is
    usually built before the Agent instance. So you always need something
    like this, which is automatically done by the agent when passing it a
    communication layer:

        self.discovery = aDiscovery

    Additionally a communication layer also requires a Messaging instance in
    order to be able to post messages in the local message queue. This
    association is automatically done when creating a Messaging instance.

        comm1 = InProcessCommunicationLayer()
        messaging = Messaging(name, comm)

    Parameters
    ----------

    on_error: str
        on_error behavior, must be refactored, not really used ATM.

    """

    def __init__(self, on_error=None) -> None:
        self._on_error = on_error
        self.discovery = None
        self.messaging = None
        self._failed_msg = defaultdict(lambda: [])

    @property
    def address(self):
        """
        An address that can be used to sent messages to this communication 
        layer.
        The concrete type of object returned depends on the class 
        implementing the CommunicationLayer protocol. 
        """
        raise NotImplementedError("Protocol class")

    def send_msg(
        self,
        src_agent: str,
        dest_agent: str,
        msg: ComputationMessage,
        on_error=None,
        from_retry=False,
    ):
        """

        Parameters
        ----------
        src_agent: str
            name of the sender agent
        dest_agent: str
            name of the target agent
        msg: ComputationMessage
            the message
        on_error:
            error handling mode, overrides the default mode set when creating
            the CommunicationLayer instance
        from_retry:
            internal arg, do NOT use.

        """
        raise NotImplementedError("Protocol class")

    def shutdown(self):
        raise NotImplementedError("Protocol class")

    def _on_send_error(self, src_agent, dest_agent, msg, on_error, exception):
        if on_error == "fail":
            raise exception(
                "Error when sending message {} -> {} : {}".format(
                    src_agent, dest_agent, msg
                )
            )
        elif on_error == "ignore":
            logger.warning(
                "could not send message from %s to %s, ignoring : " "%s",
                src_agent,
                dest_agent,
                msg,
            )
            return True
        elif on_error == "retry":
            logger.warning(
                "could not send message from %s to %s, will retry " "later : %s",
                src_agent,
                dest_agent,
                msg,
            )
            self._failed_msg[dest_agent].append((src_agent, dest_agent, msg, on_error))
            return False
        else:
            logger.warning(
                "could not send message from %s to %s, "
                "and no on_erro policy : ignoring : "
                "%s",
                src_agent,
                dest_agent,
                msg,
            )
            return False

    def retry(self, dest_agent: str):
        """
        Attempt to send all failed messages for this agent.

        :param dest_agent:
        :return:
        """
        for src, dest, msg, on_error in self._failed_msg[dest_agent]:
            logger.warning(
                "retrying delivery of message from %s to %s : " "%s", src, dest, msg
            )
            self.send_msg(src, dest, msg, on_error, from_retry=True)


class UnreachableAgent(Exception):
    pass


class InProcessCommunicationLayer(CommunicationLayer):
    """
    Implements communication for several thread-based agents in the same
    process.

    For in process communication, we don't really have an address,
    instead we directly use InProcessCommunicationLayer instances as
    addresses.

    """

    def __init__(self, on_error=None):
        super().__init__(on_error)

    @property
    def address(self):
        """
        For in-process communication, we use the object itself as the address.
        :return: 
        """
        return self

    def send_msg(
        self,
        src_agent: str,
        dest_agent: str,
        msg: ComputationMessage,
        on_error=None,
        from_retry=False,
    ):
        """
        Send a message to an agent.
        
        :param src_agent: name of the source agent 
        :param dest_agent: name of the agent
        :param msg: the message, can be any python object (only with 
        InProcess communication)
        :param on_error: how to handle failure when sending the message.
        When used, this parameter overrides the behavior set when building
        the CommunicationLayer.
        """

        on_error = on_error if on_error is not None else self._on_error
        try:
            address = self.discovery.agent_address(dest_agent)
            address.receive_msg(src_agent, dest_agent, msg)


        except UnknownAgent:
            logger.warning(
                f"Sending message from {src_agent} to unknown agent {dest_agent} :"
                f" {msg} "
            )
            return self._on_send_error(
                src_agent, dest_agent, msg, on_error, UnknownAgent
            )
        return True

    def receive_msg(self, src_agent: str, dest_agent: str, msg: ComputationMessage):
        """
        Called when receiving a message.
        
        :param src_agent: name of the source agent 
        :param dest_agent: name of the agent
        :param msg: the message, must be an iterable containing
        src_computation, dest_computation, message obejct (which can be any
        python object with InProcess communication)
        """
        src_computation, dest_computation, msg_obj, msg_type = msg
        self.messaging.post_msg(src_computation, dest_computation, msg_obj, msg_type)

    def shutdown(self):
        # There's no resources to release for InProcessCommunicationLayer as
        # message passing is implemented as simple function calls.
        pass

    # def force_get_address(self, agt_name: str):
    #     # FIXME : horrible hack until We implment a proper discovery method
    #     # This only works for in-process communication and it only works
    #     # because the InProcessCommLayer is used as the address.
    #     return self.discovery.agent_address('orchestrator')\
    #         .discovery.agent_address(agt_name)

    def __str__(self):
        return "InProcessCommunicationLayer({})".format(self.messaging)

    def __repr__(self):
        return "Comm({})".format(self.messaging)


def find_local_ip():
    # from https://stackoverflow.com/a/28950776/261821
    # public domain/free for any use as stated in comments

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


class HttpCommunicationLayer(CommunicationLayer):
    """
    This class implements the CommunicationLayer protocol.

    It uses an http server and client to send and receive messages.

    Parameters
    ----------
    address_port: optional tuple (str, int)
        The IP address and port this HttpCommunicationLayer will be
        listening on.
        If the ip address or the port are not given ,we try to use the
        primary IP address (i.e. the one with a default route) and listen on
        port 9000.

    on_error: str
        Indicates how error when sending a message will be
        handled, possible value are 'ignore', 'retry', 'fail'

    """

    def __init__(
        self,
        address_port: Optional[Tuple[str, int]] = None,
        on_error: Optional[str] = "ignore",
    ):
        super().__init__(on_error)
        if not address_port:
            self._address = find_local_ip(), 9000
        else:
            ip_addr, port = address_port
            ip_addr = ip_addr if ip_addr else find_local_ip()
            ip_addr = ip_addr if ip_addr else "0.0.0.0"
            port = port if port else 9000
            self._address = ip_addr, port

        self.logger = logging.getLogger(
            "infrastructure.communication.HttpCommunicationLayer"
        )
        self._start_server()

    def shutdown(self):
        self.logger.info("Shutting down HttpCommunicationLayer " "on %s", self.address)
        self.httpd.shutdown()
        self.httpd.server_close()

    def _start_server(self):
        # start a server listening for messages
        self.logger.info(
            "Starting http server for HttpCommunicationLayer " "on %s", self.address
        )
        try:
            _, port = self._address
            self.httpd = HTTPServer(("0.0.0.0", port), MPCHttpHandler)
        except OSError:
            self.logger.error(
                "Cannot bind http server on adress {}".format(self.address)
            )
            raise
        self.httpd.comm = self

        t = Thread(name="http_thread", target=self.httpd.serve_forever, daemon=True)
        t.start()

    def on_post_message(self, path, sender, dest, msg: ComputationMessage):
        self.logger.debug("Http message received %s - %s %s", path, sender, dest)
        self.messaging.post_msg(msg.src_comp, msg.dest_comp, msg.msg, msg.msg_type)

    @property
    def address(self) -> Tuple[str, int]:
        """
        An address that can be used to sent messages to this communication
        layer.

        :return the address as a (ip, port) tuple
        """
        return self._address

    def send_msg(
        self, src_agent: str, dest_agent: str, msg: ComputationMessage, on_error=None
    ):
        """
        Send msg from src_agent to dest_agent.

        :param src_agent:
        :param dest_agent:
        :param msg: the message to send
        :param on_error: how to handle failure when sending the message.
        When used, this parameter overrides the behavior set when building
        the HttpCommunicationLayer.
        :return:
        """
        on_error = on_error if on_error is not None else self._on_error
        try:
            server, port = self.discovery.agent_address(dest_agent)
        except UnknownAgent:
            return self._on_send_error(
                src_agent, dest_agent, msg, on_error, UnknownAgent
            )

        dest_address = "http://{}:{}/pydcop".format(server, port)
        msg_repr = simple_repr(msg.msg)
        try:
            r = requests.post(
                dest_address,
                headers={
                    "sender-agent": src_agent,
                    "dest-agent": dest_agent,
                    "sender-comp": msg.src_comp,
                    "dest-comp": msg.dest_comp,
                    "type": str(msg.msg_type),
                },
                json=msg_repr,
                timeout=0.5,
            )
        except ConnectionError:
            # Could not reach the target agent: connection refused or name
            # or service not known
            return self._on_send_error(
                src_agent, dest_agent, msg, on_error, UnreachableAgent
            )

        if r is not None and r.status_code == 404:
            # It seems that the target computation of this message is not
            # hosted on the agent
            return self._on_send_error(
                src_agent, dest_agent, msg, on_error, UnknownComputation
            )
        return True

    def __str__(self):
        return "HttpCommunicationLayer({}:{})".format(*self._address)


class MPCHttpHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        sender, dest = None, None
        type = MSG_ALGO
        if "sender-agent" in self.headers:
            sender = self.headers["sender-agent"]
        if "dest-agent" in self.headers:
            dest = self.headers["dest-agent"]
        src_comp, dest_comp = None, None
        if "sender-comp" in self.headers:
            src_comp = self.headers["sender-comp"]
        if "dest-comp" in self.headers:
            dest_comp = self.headers["dest-comp"]
        if "type" in self.headers:
            type = self.headers["type"]

        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        try:
            content = json.loads(str(post_data, "utf-8"))
        except JSONDecodeError as jde:
            print(jde)
            print(post_data)
            raise jde

        comp_msg = ComputationMessage(
            src_comp, dest_comp, from_repr(content), int(type)
        )
        try:
            self.server.comm.on_post_message(self.path, sender, dest, comp_msg)

            # Always answer 200, as the actual message is not processed yet by
            # the target computation.
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()

        except UnknownComputation as e:
            # if the requested computation is not hosted here
            self.send_response(404, str(e))
            self.send_header("Content-type", "text/plain")
            self.end_headers()

    def log_request(self, code="-", size="-"):
        # Avoid logging all requests to stdout
        pass


MSG_MGT = 10
MSG_VALUE = 15
MSG_ALGO = 20


class Messaging(object):
    """
    A `Messaging` instance is responsible for all messaged-based communication
    (sending and receiving messages) for an agent.

    Received messages a stored in a queue and can be fetched using `next_msg`.

    When sending messages, using `post_msg`, messages are dispatched
    either internally (directly to the queue) when the target is
    registered on this Messaging instance, or the actual sending is
    delegated to a CommunicationLayer instance (which implement a network
    communication protocol).

    Also accumulates metrics on messages sending.

    Parameters
    ----------
    agent_name: str
        name of the agent this Messaging instance will send message for.
    comm: CommunicationLayer
        a concrete implementation of the CommunicationLayer protocol, it will
        be used to send messages to other agents.
    delay: int
        an optional delay between message delivery, in second. This delay
        only applies to algorithm's messages and is useful when you want to
        observe (for example with the GUI) the behavior of the algorithm at
        runtime.
    """

    def __init__(self, agent_name: str, comm: CommunicationLayer, delay: float = None):
        self._queue = PriorityQueue()
        self._local_agent = agent_name
        self.discovery = comm.discovery
        self._comm = comm
        self._comm.messaging = self
        self._delay = delay
        self.logger = logging.getLogger(f"infrastructure.communication.{agent_name}")

        # Keep track of failer messages to retry later
        self._failed = []

        # Containers for metrics on sent messages:
        self.count_ext_msg = defaultdict(lambda: 0)  # type: Dict[str, int]
        self.size_ext_msg = defaultdict(lambda: 0)  # type: Dict[str, int]
        self.last_msg_time = 0
        self.msg_queue_count = 0

        self._shutdown = False

    @property
    def communication(self) -> CommunicationLayer:
        return self._comm

    @property
    def local_agent(self) -> str:
        """
        The name of the local agent.
        Returns
        -------
        The name of the agent this Messaging instance is sending messages for.
        """
        return self._local_agent

    @property
    def count_all_ext_msg(self) -> int:
        """
        Count of all non-management external messages sent.
        :return:
        """
        return sum(v for v in self.count_ext_msg.values())

    @property
    def size_all_ext_msg(self) -> int:
        """
        Size of all non-management external messages sent.
        :return:
        """
        return sum(v for v in self.size_ext_msg.values())

    def next_msg(self, timeout: float = 0):
        try:
            msg_type, _, t, full_msg = self._queue.get(block=True, timeout=timeout)
            if self._delay and msg_type == MSG_ALGO:
                sleep(self._delay)
            return full_msg, t
        except Empty:
            return None, None

    def post_msg(
        self,
        src_computation: str,
        dest_computation: str,
        msg,
        msg_type: int = MSG_ALGO,
        on_error=None,
    ):
        """
        Send a message `msg` from computation `src_computation` to computation
        `dest_computation`.

        Messages can be sent
          * either to one of our local computations
          * or to a computation hosted on another agent
        If the message is for a local computation, deliver it directly,
        otherwise, we delegate to he communication layer.

        Notes
        -----

        priority level : messages are sent with a priority level.

        If the agent hosting the target computation is not known, we will
        retry sending as soon as the hosting is registered. There is
        currently no time limit for this, meaning that a message can stay in
        fail state forever and never been delivered if the corresponding
        computation is never registered.

        TODO: implement some kind of timeout mechanism to report an error if
        message stay in the failed stay for too long.

        Parameters
        ----------
        src_computation: str
            name of the computation sending the messages
        dest_computation: str
            name of the computation the message is sent to.
        msg: the message
        msg_type: int
            the type of the message, like MSG_ADM or MSG_ALGO. Defaults to
            MSG_ALGO. Used to send messages with an higher priority first.
        on_error: ??
        """
        if self._shutdown:
            return

        msg_type = MSG_ALGO if msg_type is None else msg_type
        try:
            dest_agent = self.discovery.computation_agent(dest_computation)
        except UnknownComputation:
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(
                    f"Cannot send msg from {src_computation} to unknown "
                    f"comp {dest_computation}, will retry  later : {msg}"
                )
            self.discovery.subscribe_computation(
                dest_computation, self._on_computation_registration, one_shot=True
            )
            self._failed.append(
                (src_computation, dest_computation, msg, msg_type, on_error)
            )
            return

        full_msg = ComputationMessage(src_computation, dest_computation, msg, msg_type)
        if dest_agent == self._local_agent:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Posting local message {src_computation} -> "
                    f"{dest_computation} : {msg}"
                )
            now = perf_counter()
            if msg_type != MSG_MGT:
                self.last_msg_time = now
            # When putting the message in the queue we add the type,
            # a monotonic msg counter and the time of reception. As the queue
            # is a priority queue, putting type and counter first ensure
            # that the #  tuple will always be orderable. The time is
            # useful to measure the delay between reception and handling
            # of a message.
            self.msg_queue_count += 1
            self._queue.put((msg_type, self.msg_queue_count, now, full_msg))
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Posting remote message {src_computation} -> "
                    f"{dest_computation} : {msg}"
                )

            # If the destination is on another agent, it means that the
            # message source must be one of our local computation and we
            # should know about it.

            # NOTE: the computation might have been removed, but that's considered as a
            # bug, a computation should not send message once removed
            try:
                self.discovery.computation_agent(src_computation)
            except:
                self.logger.error(f"Could not find src computation {src_computation} "
                                  f" when posting msg {msg} to {dest_computation} "
                                  f"{dest_agent}, {self._local_agent}) ")
                raise

            # send using Communication Layer
            if msg_type != MSG_MGT:
                self.count_ext_msg[src_computation] += 1
                self.size_ext_msg[src_computation] += msg.size

            self._comm.send_msg(
                self._local_agent, dest_agent, full_msg, on_error=on_error
            )

    def shutdown(self):
        """Shutdown messaging

        No new message will be sent and any new message posted will be
        silently dropped.
        However it is still possible to call ``next_msg`` to empty the queue
        and handle all message received before ``shutdown` was called.
        """
        self._shutdown = True

    def _on_computation_registration(self, evt: str, computation: str, agent: str):
        """
        Callback for DeploymentInfo on computatino registration.

        Called when a new computation-agent is registered.
        """

        if evt == "computation_added":
            for failed in self._failed[:]:
                src, dest, msg, msg_type, on_error = failed
                if dest != computation:
                    continue
                self.logger.info(
                    "Retrying failed message to %s on %s : %s", dest, agent, msg
                )
                self.post_msg(src, dest, msg, msg_type, on_error)
                self._failed.remove(failed)

    def __str__(self):
        return "Messaging({})".format(self._local_agent)
