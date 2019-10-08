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
The ``computations`` module contains base classes for computation and message.

You generally need to sub-class classes from this module when implementing
your own DCOP algorithm.

"""


import logging
from functools import wraps
from importlib import import_module
from typing import List, Tuple, Any, Callable, Dict, Optional

from numpy import random

from pydcop.algorithms import ComputationDef, load_algorithm_module
from pydcop.dcop.objects import Variable
from pydcop.utils.simple_repr import SimpleRepr, SimpleReprException, simple_repr
from pydcop.infrastructure.Events import event_bus


class Message(SimpleRepr):
    """
    Base class for messages.

    you generally sub-class ``Message`` to define the message type for a DCOP
    algorithm.
    Alternatively you can use :py:func:`message_type` to create
    your own message type.


    Parameters
    ----------
    msg_type: str
       the message type ; this will be used to select the correct handler
       for a message in a DcopComputation instance.
    content: Any
       optional, usually you sub-class Message and add your own content
       attributes.

    """

    def __init__(self, msg_type, content=None):
        self._msg_type = msg_type
        self._content = content

    @property
    def size(self):
        """
        Returns the size of the message.

        You should overwrite this methods in subclasses,
        will be used when computing the communication load of an
        algorithm and by some distribution methods that optimize
        the distribution of computation for communication load.

        Returns
        -------
        size : int

        """
        return 0

    @property
    def type(self) -> str:
        """
        The type of the message.

        Returns
        -------
        message_type: str
        """
        return self._msg_type

    @property
    def content(self):
        return self._content

    def __str__(self):
        return f"Message({self.type})"

    def __repr__(self):
        return f"Message({self.type}, {self.content})"

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return self.type == other.type and self.content == other.content


def message_type(msg_type: str, fields: List[str]):
    """
    Class factory method for Messages

    This utility method can be used to easily define new Message type without
    subclassing explicitly (and manually) the Message class. Tt output a
    class object which subclass Message.

    Message instance can be created from the return class type using either
    keywords arguments or positional arguments (but not both at the same time).

    Instances from Message classes created with `message_type` support
    equality, simple_repr and have a meaningful str representation.

    Parameters
    ----------
    msg_type: str
        The type of the message, this will be return by `msg.type` (see example)
    fields: List[str]
        The fields in the message

    Returns
    -------
    A class type that can be used as a message type.

    Example
    -------

    >>> MyMessage = message_type('MyMessage', ['foo', 'bar'])
    >>> msg1 = MyMessage(foo=42, bar=21)
    >>> msg = MyMessage(42, 21)
    >>> msg.foo
    42
    >>> msg.type
    'MyMessage'
    >>> msg.size
    0
    """

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError("Use positional or keyword arguments, but not " "both")
        if args:
            if len(args) != len(fields):
                raise ValueError("Wrong number of positional arguments")
            for f, a in zip(fields, args):
                setattr(self, f, a)

        for k, v in kwargs.items():
            if k not in fields:
                raise ValueError("Invalid field {k} in {msg_type}")
            setattr(self, k, v)
        Message.__init__(self, msg_type, None)

    def to_str(self):
        fs = ", ".join([f + ": " + str(getattr(self, f)) for f in fields])
        return msg_type + "(" + fs + ")"

    def _simple_repr(self):

        # Full name = module + qualifiedname (for inner classes)
        r = {
            "__module__": self.__module__,
            "__qualname__": "message_type",
            "__type__": self.__class__.__qualname__,
        }
        for arg in fields:
            try:
                val = getattr(self, arg)
                r[arg] = simple_repr(val)
            except AttributeError:
                if hasattr(self, "_repr_mapping") and arg in self._repr_mapping:
                    try:
                        r[arg] = self.__getattribute__(self._repr_mapping[arg])
                    except AttributeError:
                        SimpleReprException(
                            f"Invalid repr_mapping in {self}, "
                            "no attribute for {self._repr_mapping[arg]}"
                        )

                else:
                    raise SimpleReprException(
                        "Could not build repr for {self}, " "no attribute for {arg}"
                    )
        return r

    def equals(self, other):
        if self.type != other.type:
            return False
        if self.__dict__ != other.__dict__:
            return False
        return True

    msg_class = type(
        msg_type,
        (Message,),
        {
            "__init__": __init__,
            "__str__": to_str,
            "__repr__": to_str,
            "_simple_repr": _simple_repr,
            "__eq__": equals,
        },
    )
    return msg_class


class ComputationException(Exception):
    """
    Base class for exception from computations.
    """

    pass


class ComputationMetaClass(type):
    """
    ComputationMetaClass is used to ensure that each subclass of
    `MessagePassingComputation` has it's own set of message handlers,
    which can be declared using the `@register` decorator.

    See Also
    --------

    register(message_type) decorator.
    """

    def __new__(mcs, clsname, bases, attrs):
        cls = super().__new__(mcs, clsname, bases, attrs)
        # Each class using this metaclass must have it's own set of handlers
        cls._decorated_handlers = {}
        for attr in attrs.values():
            # handlers registered using the decorator have a specific msg_type
            # attribute and must be added to the dict of handlers.
            if hasattr(attr, "msg_type"):
                cls._decorated_handlers[attr.msg_type] = attr
        return cls


class MessagePassingComputation(object, metaclass=ComputationMetaClass):
    """
    `MessagePassingComputation` is the base class for all computations.
    It defines the computation lifecycle (`start`, `pause`, `stop`) and can
    send and receive messages.

    When subclassing `MessagePassingComputation`, you can use the `@register`
    decorator on your subclass methods to register then as handler for a
    specific kind of message. For example::

    > class MyComputation(MessagePassingComputation):
    >    ...
    >    @register('msg_on_c')
    >    def handler_c(self, s, m, t):
    >        print("received messages", m, "from", s)

    Note
    ----
    A computation is always be hosted and run on an agent, which works with a
    single thread. This means that its methods do not need to be thread safe
    as they will always be called sequentially in the same single thread.


    Parameters
    ----------
    name: str
        The name of the computation.

    """

    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name
        self._msg_handlers = {}
        # Default logger for computation, will generally be overwritten by
        # sub-classes.
        self.logger = logging.getLogger("pydcop.computation." + self.__class__.__name__)

        self._msg_sender = None
        self._periodic_action_handler = None
        self._running = False

        self._is_paused = False
        self._paused_messages_post = []  # type: List[Tuple[str, Any, int, Any]]
        self._paused_messages_recv = []  # type: List[Tuple[str, Any, float]]

    @property
    def name(self) -> str:
        """
        An computation must have a name, which is used as a target address
        when sending messages to the algorithm.

        Returns
        -------
        str
            The name of the computation
        """
        return self._name

    @property
    def is_running(self) -> bool:
        """
        bool: True if the computation is currently running
        """
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def message_sender(self) -> Callable[[str, str, Message, int], None]:
        return self._msg_sender

    @message_sender.setter
    def message_sender(self, msg_sender: Callable[[str, str, Message, int], None]):
        if self._msg_sender is not None:
            raise AttributeError("Can only set message sender once ")
        self._msg_sender = msg_sender

    @property
    def periodic_action_handler(self):
        return self._periodic_action_handler

    @periodic_action_handler.setter
    def periodic_action_handler(self, handler):
        if self._periodic_action_handler is not None:
            raise AttributeError("Can only set periodic_action_handler once")
        self._periodic_action_handler = handler

    def finished(self):
        pass

    def start(self):
        """
        Start the computation.

        A computation will only handle messages once it has been started.
        However, due to the asynchronous and distributed nature of the
        system, computation may send messages to computations that did not
        start yet ; any message received before startup is kept and handled once
        `start()` has been called.

        This method should not be overwritten in subclasses, instead you can
        overwrite the `on_start` hook, which is called automatically when the
        computation starts, before handling pending messages received before
        startup.

        """
        self._running = True
        self.on_start()

        pending_msg_count = 0
        while self._paused_messages_recv:
            pending_msg_count += 1
            src, msg, t = self._paused_messages_recv.pop()
            # Do NOT call on_message directly, that would block the
            # agent's thread for a potentially long time during which we
            # would not be able to handle any mgt message.
            # Instead, inject the message with a slightly higher
            # priority so that we handle them before new messages.
            self._msg_sender(src, self.name, msg, 19)
        self.logger.debug(
            f"On starting {self.name}, injecting {pending_msg_count} pending messages"
            f" received before start"
        )

    def stop(self):
        """
        Stop the computation.

        This method should not be overwritten in subclasses, instead you can
        overwrite the `on_stop` hook, which is called automatically when the
        computation stops.
        """
        self.logger.info(f"Stopping computation {self._name}")
        self._running = False
        self.on_stop()

    def pause(self, is_paused: bool = True):
        """
        Pauses or resumes the computation.

        Notes
        -----
        When in pause, a computation does not send nor handle any messages.
        Messages are instead stored to be handled, or posted, once the
        computation is resumed.

        When resuming a paused computation, any stored messages is handled
        or sent.

        Parameters
        ----------
        is_paused: bool
            requested pause state for the computation.
        """
        if self._is_paused != is_paused:
            self._is_paused = is_paused
            self.on_pause(is_paused)

        if not is_paused:

            waiting_msg_count = 0
            while self._paused_messages_post:
                waiting_msg_count += 1
                target, msg, prio, e = self._paused_messages_post.pop()
                self.post_msg(target, msg, prio, e)
            self.logger.debug(
                "On resume, posting %s pending messages ", waiting_msg_count
            )

            waiting_msg_count = 0
            while self._paused_messages_recv:
                waiting_msg_count += 1
                src, msg, t = self._paused_messages_recv.pop()
                # Do NOT call on_message directly, that would block the
                # agent's thread for a potentially long time during which we
                # would not be able to handle any mgt message.
                # Instead, inject the message back with a slightly higher
                # priority so that we handle them before new messages.
                self._msg_sender(src, self.name, msg, 19)
            self.logger.debug(
                "On resume, re-injecting %s received pending " "messages ",
                waiting_msg_count,
            )

    def on_start(self):
        """
        Called when starting the computation.

        This method is meant to be overwritten in subclasses.
        """
        pass

    def on_stop(self):
        """
        Called when stopping the computation

        This method is meant to be overwritten in subclasses.
        """
        pass

    def on_pause(self, paused: bool):
        """
        Called when pausing or resuming the computation.

        This method is meant to be overwritten in subclasses.

        Parameters
        ----------
        paused: boolean
            the new pause status. This method is only called is the status has changed

        """
        pass

    def on_message(self, sender: str, msg: Message, t: float):
        """
        This method is called by the hosting agent when a message for this
        computation object has been received.

        Notes
        -----
        Subclasses of :class:`MessagePassingComputation` should not override
        this method, as it is used for storing received messages during pause.
        Instead, they should register a method for each message type by using
        the :func:`register` decorator.

        Parameters
        ----------
        sender: str
            the name of the computation that has sent this
        msg: an instance of Message
            the received message that must be handled
        t: float
            reception time
        """

        if not self.is_paused and self._running:
            # Only handle messages if the computation has been started and is
            # not paused.
            event_bus.send(
                "computations.message_rcv." + self.name, (self.name, msg.size)
            )
            try:
                self._decorated_handlers[msg.type](self, sender, msg, t)
            except KeyError:
                self._msg_handlers[msg.type](sender, msg, t)
        else:
            self.logger.debug(
                f"Storing message from {sender} to {self.name} {msg} . "
                f"paused {self.is_paused}, running {self._running}"
            )
            self._paused_messages_recv.append((sender, msg, t))

    def post_msg(self, target: str, msg, prio: int = None, on_error=None):
        """
        Post a message.

        Notes
        -----
        This method should always be used when sending a message. The
        computation should never use the `_msg_sender` directly has this
        would bypass the pause state.

        Parameters
        ----------
        target: str
            the target computation for the message
        msg: an instance of Message
            the message to send
        prio: int
            priority level
        on_error: error handling method
            passed to the messaging component.
        """
        if not self.is_paused:
            self._msg_sender(self.name, target, msg, prio, on_error)
            event_bus.send(
                "computations.message_snd." + self.name, (self.name, msg.size)
            )
        else:
            self._paused_messages_post.append((target, msg, prio, on_error))

    def add_periodic_action(self, period: float, cb: Callable):
        """
        Add an action that will be called every `period` seconds.

        Parameters
        ----------
        period: float
            the period, in second
        cb: Callable
            A callable, with no parameters

        Returns
        -------
        the callable
        """

        def call_action():
            if not self.is_paused:
                cb()

        return self.periodic_action_handler.set_periodic_action(period, call_action)

    def remove_periodic_action(self, handle):
        self.periodic_action_handler.remove_periodic_action(handle)

    def __repr__(self):
        return "MessagePassingComputation({})".format(self.name)


# noinspection PyPep8Naming
class register(object):
    """
    Decorator for registering message handles in computations.

    This decorator is meant to be used to register message handlers in
    computation class (subclasses of MessagePassingComputation).

    A method decorated with `@register('foo')` will be called automatically
    when the computation receives a message of type `'foo'`.

    Message handler must accept 3 parameters : (sender_name, message, time).
    * sender_name is the name of the computation that sent the message
    * message is the message it-self, which is an instance of a subclass of
    `Message`
    * time is the time the message was received.

    Parameters
    ----------
    msg_type: str
        the type of message

    Example
    -------
    ::

    > class C(MessagePassingComputation):
    >    ...
    >    @register('msg_on_c')
    >    def handler_c(self, s, m, t):
    >        print("received messages", m, "from", s)

    See DsaTuto sample implementation for a complete example.

    """

    def __init__(self, msg_type: str):
        self.msg_type = msg_type

    def __call__(self, handler):
        @wraps(handler)
        def wrapper(*args, **kwargs):
            return handler(*args, **kwargs)

        wrapper.msg_type = self.msg_type
        return wrapper


class SynchronizationMsg(Message):
    def __init__(self):
        super().__init__("cycle_sync")
        # cycle_id is set by the `SynchronousComputationMixin` when sending the message.
        self.cycle_id = None

    def __repr__(self):
        return f"SynchronizationMsg()"


class SynchronousComputationMixin:
    """
    This mixin can be used with `MessagePassingComputation` classes (and classes
    deriving from it) and implements synchronicity for these computation.

    A computations that uses the `SynchronousComputationMixin` is a synchronous
    computation that respects the synchronous network model (see Distributed
    Algorithms); these computations operate in rounds: at each round i,
    a computation collects messages sent at i-1 and send messages that will be
    received at i+1.

    Due to method resolution order, the mixin MUST be declared first in the list
    of classes your are deriving from:

    >   class MyComp( SynchronousComputationMixin,  MessagePassingComputation):
    >       ...

    A synchronous computation must not handle its message directly on the message
    handler registered with @register, instead it must handle them when the
    on_new_cycle is called.
    However, this decorator must still be used in order to declare the message
    types supported by the computation, but the bodu of the method should be empty:

    >   class C( SynchronousComputationMixin,  MessagePassingComputation):
            ...
            @register("foo")
            def on_foo_msg(self, s, m , t):
                pass


    Note : the class this mixin is applied to MUST have a `neighbors` property that
    returns a list containing the name of its neighbors (see `DcopComputation` for
    example).

    """

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._current_cycle = 0
        self._cycle_messages = {}
        self._next_cycle_messages = {}
        self.cycle_message_sent = []
        # Handlers are declared at the class level in MessagePassingComputation
        # But to count cycles we must operate at the class level. For this reason
        # we override the class-level `_decorated_handlers` map with an instance-level
        # one.
        self._decorated_handlers = {"cycle_sync": self._sync_message_handler}

        for msg_type, handler in self.__class__._decorated_handlers.items():
            self._decorated_handlers[msg_type] = self._sync_message_handler

    def _sync_message_handler(self, _, sender, msg, t):
        if sender not in self.neighbors:
            raise ComputationException(
                f"Invalid message: received a message from {sender}, which is "
                f"not in the neighbors list: {self.neighbors}"
            )

        # We might have a difference of at most one cycle with our neighbors,
        # meaning we might receive messages for cycle i+1 while we are still
        # at cycle i, waiting for messages from other neighbors. In that case,
        # we simply store these messages for the next cycle.
        # If the difference is more that one cycle, there's a bug !
        if msg.cycle_id == self._current_cycle:

            if sender in self._cycle_messages:
                # We could allow several messages from a single neighbor
                # in a cycle, but that's more complicated
                # (would require grouping them for example) and I don't see this
                # as very useful. For now, simply forbid it.
                raise ComputationException(
                    f"Invalid message, {self.name} received two messages "
                    f"from {sender} for cycle {self._current_cycle}. "
                    f"In a synchronized computation, a neighbor can only send "
                    f"a single message in a cycle."
                )

            self._cycle_messages[sender] = (msg, t)

            # Check if end of cycle, Call on cycle.
            if len(self._cycle_messages) == len(self.neighbors):
                self._switch_cycle()
            else:
                self.logger.debug(f"on message from {sender}, cycle {self._current_cycle} not finished {self._cycle_messages} != {self.neighbors}")
        elif msg.cycle_id == self._current_cycle + 1:
            self._next_cycle_messages[sender] = (msg, t)
        else:
            raise ComputationException(
                f"Invalid message for computation {self.name}, "
                f"current cycle is {self._current_cycle} "
                f"but received message for cycle {msg.cycle_id} "
                f"from {sender}"
            )

    @property
    def current_cycle(self):
        return self._current_cycle

    def post_msg(self, target: str, msg, prio: int = None, on_error=None):
        # We need to add the current cycle_id to all messages, in order for the neighbor
        # to be able to check that the message is for the current or next cycle.
        self.logger.debug(
            f"Sending msg for cycle {self._current_cycle} {self.name} -> {target} : {msg}"
        )
        msg.cycle_id = self._current_cycle
        super(SynchronousComputationMixin, self).post_msg(target, msg, prio, on_error)
        self.cycle_message_sent.append(target)

    def start(self):
        super(SynchronousComputationMixin, self).start()

        # Startup (on_start handler) is considered to be the cycle 0.
        # After this cycle 0, send a synchronization message to all neighbors
        # to which we did not already send a algo-level message.
        for neighbor in list(self.neighbors):
            # Some messages might also have been sent using post_msg
            if neighbor not in self.cycle_message_sent:
                self.post_msg(neighbor, SynchronizationMsg())

        self._cycle_messages = self._next_cycle_messages
        self._next_cycle_messages = {}

    def _switch_cycle(self):
        self.logger.debug(f"Running cycle {self._current_cycle}")
        self._current_cycle += 1
        algo_message = {
            k: (msg, t)
            for k, (msg, t) in self._cycle_messages.items()
            if not isinstance(msg, SynchronizationMsg)
        }
        self.cycle_message_sent = []
        messages = self.on_new_cycle(algo_message, self._current_cycle - 1)

        # For synchronization, we need to send messages to _all_ neighbors, even this
        # implemented algorithms does not require some (or in many cases, most) of these
        # messages.
        remaining_neighbors = list(self.neighbors)
        if messages is not None:
            for target, message in messages:
                # message.cycle_id = self._current_cycle
                self.post_msg(target, message)
                remaining_neighbors.remove(target)
        self.logger.debug(f"After cycle {self.current_cycle-1}, need to send sync msg to {remaining_neighbors}")

        # Now send a cycle synchronization message to all neighbors to which we did not
        # already send a algo-level message.
        for neighbor in remaining_neighbors:
            # Some messages might also have been sent using post_msg
            if neighbor not in self.cycle_message_sent:
                self.logger.debug(
                    f"After cycle {self.current_cycle - 1}, sync msg to {neighbor}")

                self.post_msg(neighbor, SynchronizationMsg())

        self._cycle_messages = self._next_cycle_messages
        self._next_cycle_messages = {}

    @property
    def cycle_count(self):
        return self._current_cycle

    def on_new_cycle(self, messages: Dict[str, Tuple], cycle_id) -> Optional[List]:
        """
        Called when switching to a new cycle.

        This method must be overridden in derived classes, this is where you write the
        logic of your synchronous algorithm. It is automatically called when
        all messages have been received for the current cycle and should be handled.
        When handling these messages, the computation will generally also send messages
        to some of its neighbors; this can be achieved either by using `post_msg` or
        by returning the list of messages that must be sent for this cycle. Notice that
        these messages will be delivered in the next cycle.

        Notes
        -----
        You are only allowed to send at most one message to any given neighbor in a
        cycle. If you send several messages to the same neighbor, a ComputationException
        will be raised by this neighbor in the next cycle, when receiving them.

        Parameters
        ----------
        messages: dict
            all the messages received for this cycle (i.e. that have been set by
            neighbors during the previous cycle). The keys of the dict are the sender
            of the message, the values are tuple (message, time).
        cycle_id: int
            id for this cycle

        Returns
        -------
        messages: list of None
            a list of tuples (target, message) with the messages to be sent to neighbors
            for this cycle (which they will receive at the next cycle).

        """
        #
        pass


class DcopComputation(MessagePassingComputation):
    """
    Computation representing a DCOP algorithm.

    DCOP algorithm computation are like base computation : they work by
    exchanging messages with each other. The only difference is that a DCOP
    computation has additional metadata representing the part of the DCOP it
    as been defined for.

    Parameters
    ----------
    name: string
        the name of the computation
    comp_def: ComputationDef
        the ComputationDef object contains the information about the dcop
        computation : algorithm used, neighbors, etc.

    Notes
    -----

    When subclassing DcopComputation, you **must** declare in __init__ the
    message handlers for the message's type(s) of your algorithm. For example
    with the following ``_on_my_msg`` will be called for each incoming
    ``my-msg`` message.

        self._msg_handlers['my_msg'] = self._on_my_msg


    """

    def __init__(self, name, comp_def: ComputationDef):
        if comp_def is None or name is None:
            raise ValueError(
                "ComputationDef and name are mandatory for a DCOP " "computation"
            )
        super().__init__(name)

        self.algo_name = self.__class__.__module__.split(".")[-1]
        self.logger = logging.getLogger(f"pydcop.algo.{self.algo_name}.{name}")

        self.computation_def = comp_def
        self.__cycle_count__ = 0

    @property
    def neighbors(self) -> List[str]:
        """
        The neighbors of this computation.

        Notes
        -----
        This is just a convenience shorthand to the neighbors of the node in the
        computation node :

             my_dcop_computation.computation_def.node.neighbors

        Returns
        -------
        list of string
            a list containing the names of the neighbor computations.
        """
        return list(self.computation_def.node.neighbors)

    @property
    def cycle_count(self):
        return self.__cycle_count__

    def footprint(self):
        """
        The footprint of the computation.

        A DCOP computation has a footprint, which represents the amount of
        memory this computation requires to run. This depends on the
        algorithm used and thus must be overwritten in subclasses.
        This footprint is used when distributing computation on agents,
        to ensure that agents only host computation they can run.

        Returns
        -------
        float:
            the footprint
        """
        raise NotImplementedError()

    def new_cycle(self):
        """
        For algorithms that have a concept of cycle, you must call this
        method (in the subclass) at the start of every new cycle.
        This is used to generate statistics by cycles.

        Notes
        -----
        You can just ignore this if you do not care about cycles.

        """
        self.__cycle_count__ += 1
        self._on_new_cycle(self.cycle_count)
        event_bus.send("computations.cycle." + self.name, (self.name, self.cycle_count))

    def _on_new_cycle(self, count):
        """
        Will be called for every new cycle.

        Notes
        -----
        This method is meant to be monkey-patched when we needed to observe
        cycles. You should generally not use it nor overwrite it.

        Parameters
        ----------
        count: int
            the cycle count.
        """
        pass

    def post_to_all_neighbors(self, msg, prio: int = None, on_error=None):
        """
        Post a message to all neighbors of the computation.

        Parameters
        ----------
        msg: an instance of Message
            the message to send
        prio: int
            priority level
        on_error: error handling method
            passed to the messaging component.

        """
        for neighbor in self.neighbors:
            self.post_msg(neighbor, msg, prio, on_error)

    def __repr__(self):
        return "{}.{}({})".format(self.algo_name, self.__class__.__name__, self.name)


class VariableComputation(DcopComputation):
    """
    A VariableComputation is a dcop computation that is responsible for
    selecting the value of a variable.

    See Also
    --------
    DcopComputation


    """

    def __init__(self, variable: Variable, comp_def: ComputationDef):
        if variable is None:
            raise ValueError(
                "Variable object is mandatory for a " "VariableComputation"
            )
        if comp_def is None:
            raise ValueError(
                "ComputationDef object is mandatory for a " "VariableComputation"
            )
        super().__init__(variable.name, comp_def)
        self._variable = variable
        self.__value__ = None  # NEVER access this directly
        self.__cost__ = None  # NEVER access this directly
        self._previous_val = None

        self._footprint_method = None  # cache ref to avoid multiple imports

    @property
    def variable(self) -> Variable:
        """
        Variable:
            The variable this algorithm is responsible for, if any.
        """
        return self._variable

    @property
    def current_value(self):
        """
        Return the value currently selected by the algorithm.

        If a computation algorithm does no select a value, override this
        method to raise an exception.
        :return:
        """
        return self.__value__

    @property
    def current_cost(self):
        return self.__cost__

    def footprint(self) -> float:
        """
        Return the footprint of the computation.

        The footprint is used by many distribution methods.

        Notes
        -----

        This methods should **NOT be overwritten** when subclassing
        `VariableComputation`, instead, you should provide a
        `computation_memory` function at module-level in your algorithm.
        This method computes the footprint
        from a ComputationNode, which is required to for bootstrap
        distributions, where the distribution is computed before instanciating
        the computation objects.

        This module level function must have the following signature:

        > computation_memory(computation: ComputationNode) -> float:

        Returns
        -------
        float:
            The footprint of the computation.
        """
        if self._footprint_method is None:
            try:
                self._footprint_method = import_module(
                    self.__class__.__module__
                ).computation_memory
            except AttributeError:
                # if the algorithm as been imported without using
                # `load_algoorithm_module`, computation_memory may not be
                #  available
                self._footprint_method = lambda *a, **ka: 1

        return self._footprint_method(self.computation_def.node)

    def value_selection(self, val, cost=0):
        """
        When the computation selects a value, it MUST be done by calling
        this method. This is necessary to be able to automatically monitor
        value changes.
        :param val:
        :param cost:
        """
        if val != self._previous_val:
            self.logger.info(
                "Selecting new value: %s, %s (previous: %s, %s)",
                val,
                cost,
                self._previous_val,
                self.__cost__,
            )
            self._on_value_selection(val, cost, self.cycle_count)
            self._previous_val = val
            self.__value__ = val
            event_bus.send("computations.value." + self.name, (self.name, val))
        self.__cost__ = cost

    def random_value_selection(self):
        """
        Select a random value from the domain of the variable of the
        VariableComputation.

        """
        value = random.choice(self.variable.domain)
        self.value_selection(value)

    def _on_value_selection(self, val, cost, cycle_count):
        pass


class ExternalVariableComputation(DcopComputation):
    """
    Computation representing an external variable.

    An external variable is a variable that we cannot select a value for ;
    i.e. it is not a decision variable, its value is controlled by an
    external mechanism.
    External variables can be used to represent the environment-sensing
    aspect of an agent.

    As we cannot select a value for an external variable,the role of an
    ExternalVariableComputation instance, is simply to provide an API
    to change the value from 'outside' of the DCOp system and to
    notify other computations of this change.

    """

    def __init__(self, external_var, msg_sender=None, comp_def=None):
        super().__init__(external_var.name, comp_def)
        self._msg_handlers["SUBSCRIBE"] = self._on_subscribe_msg

        self._external_var = external_var.clone()
        self._msg_sender = msg_sender
        self.subscribers = set()

        self._external_var.subscribe(self._on_variable_change)

    @property
    def name(self):
        return self._external_var.name

    @property
    def current_value(self):
        return self._external_var.value

    def on_start(self):
        return {}

    def _on_variable_change(self, _):
        self._fire()

    def _on_subscribe_msg(self, var_name, _, t):
        self.subscribers.add(var_name)
        self._msg_sender.post_msg(
            self.name, var_name, Message("VARIABLE_VALUE", self._external_var.value)
        )

    def change_value(self, value):
        self._external_var.value = value

    def _fire(self):
        for s in self.subscribers:
            self._msg_sender.post_msg(
                self.name, s, Message("VARIABLE_VALUE", self._external_var.value)
            )

    def __str__(self):
        return "External variable computation for " + self._external_var.name

    def __repr__(self):
        return "External variable computation for " + self._external_var.name


def build_computation(comp_def: ComputationDef) -> MessagePassingComputation:
    """
    Build a concrete computation instance from a computation definition.

    :param comp_def: the computation definition
    :return: a concrete MessagePassingComputation
    """
    algo_module = load_algorithm_module(comp_def.algo.algo)
    computation = algo_module.build_computation(comp_def)
    return computation
