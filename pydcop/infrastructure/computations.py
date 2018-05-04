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
from typing import List, Tuple, Any, Callable

from pydcop.algorithms import ComputationDef
from pydcop.dcop.objects import Variable
from pydcop.utils.simple_repr import SimpleRepr, SimpleReprException, \
    simple_repr


class Message(SimpleRepr):

    def __init__(self, msg_type, content=None):
        self._msg_type = msg_type
        self._content = content

    @property
    def size(self):
        return 0

    @property
    def type(self):
        return self._msg_type

    @property
    def content(self):
        return self._content

    def __str__(self):
        return 'Message({})'.format(self.type)

    def __repr__(self):
        return 'Message({}, {})'.format(self.type, self.content)

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
    A class type that can be used as a message type

    Examples
    --------

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

    def __init__(self,  *args, **kwargs):
        if args and kwargs:
            raise ValueError('Use positional or keyword arguments, but not '
                             'both')
        if args:
            if len(args) != len(fields):
                raise ValueError('Wrong number of positional arguments')
            for f, a in zip(fields, args):
                setattr(self, f, a)

        for k, v in kwargs.items():
            if k not in fields:
                raise ValueError('Invalid field {} in {}'.format(k, msg_type))
            setattr(self, k, v)
        Message.__init__(self, msg_type, None)

    def to_str(self):
        fs = ', '.join([f + ': ' + str(getattr(self, f)) for f in fields])
        return msg_type + '(' + fs + ')'

    def _simple_repr(self):

        # Full name = module + qualifiedname (for inner classes)
        r = {'__module__': self.__module__,
             '__qualname__': 'message_type',
             '__type__': self.__class__.__qualname__}
        for arg in fields:
            try:
                val = getattr(self, arg)
                r[arg] = simple_repr(val)
            except AttributeError:
                if hasattr(self, '_repr_mapping') and arg in \
                        self._repr_mapping:
                    try:
                        r[arg] = self.__getattribute__(
                            self._repr_mapping[arg])
                    except AttributeError:
                        SimpleReprException('Invalid repr_mapping in {}, '
                                            'no attribute for {}'.
                                            format(self,
                                                   self._repr_mapping[arg]))

                else:
                    raise SimpleReprException('Could not build repr for {}, '
                                              'no attribute for {}'.
                                              format(self, arg))
        return r

    def equals(self, other):
        if self.type != other.type:
            return False
        if self.__dict__ != other.__dict__:
            return False
        return True

    msg_class = type(msg_type, (Message,),
                     {'__init__': __init__,
                      '__str__': to_str,
                      '__repr__': to_str,
                      '_simple_repr': _simple_repr,
                      '__eq__': equals
                      })
    return msg_class


class MessagePassingComputation(object):
    """

    Notes
    -----
    A computation is always be hosted, and run, on an agent, which works with a
    single thread.

    Parameters:
    name: str
        The name of the computation.
    """
    def __init__(self, name: str):
        self._name = name
        self._msg_handlers = {}
        # Default logger for computation, will generally be overwritten by
        # sub-classes.
        self.logger = logging.getLogger('pydcop.computation')
        self._msg_sender = None
        self._running = False

        self._is_paused = False
        self._paused_messages_post = []  # type: List[Tuple[str, Any, int]]
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
    def is_running(self)-> bool:
        """
        bool: True if the computation is currently running
        """
        return self._running

    @property
    def is_paused(self)-> bool:
        return self._is_paused

    @property
    def message_sender(self) -> Callable[[str, str, Message, int], None]:
        return self._msg_sender

    @message_sender.setter
    def message_sender(
            self, msg_sender: Callable[[str, str, Message, int], None]):
        if self._msg_sender is not None:
            raise AttributeError('Can only set message sender once ')
        self._msg_sender = msg_sender

    def finished(self):
        pass

    def start(self):
        self._running = True
        self.on_start()

    def stop(self):
        self.on_stop()
        self._running = False

    def pause(self, is_paused: bool=True):
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
        self._is_paused = is_paused
        if not is_paused:

            waiting_msg_count = 0
            while self._paused_messages_post:
                waiting_msg_count += 1
                target, msg, prio = self._paused_messages_post.pop()
                self.post_msg(target, msg, prio)
            self.logger.debug('On resume, posting %s pending messages ',
                             waiting_msg_count)

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
            self.logger.debug('On resume, re-injecting %s received pending '
                              'messages ', waiting_msg_count)

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
        """
        pass

    def on_message(self, sender: str, msg: Message, t: float):
        """
        This method is called by the hosting agent when a message for this
        computation object has been received.

        Notes
        -----
        Subclasses of AbstractMessagePassingAlgorithm should not override
        this method, as it is used for storing received messages during pause.
        Instead, they should register a method for each message type in
        `_msg_handlers`.

        Parameters
        ----------
        sender_name: str
            the name of the computation that has sent this
        msg: an instance of Messsage
            the received message that must be handled
        t: float
            reception time
        """
        if not self.is_paused:
            self._msg_handlers[msg.type](sender, msg, t)
        else:
            self._paused_messages_recv.append((sender, msg, t))

    def post_msg(self, target: str, msg, prio: int=None):
        """
        Post a message.

        Notes
        -----
        This method should always be used when seinding a message. The
        computation should never use the `_msg_sender` directly has this
        would bypass the pause state.

        Parameters
        ----------
        target: str
            the target computation for the message
        msg: an instance of Message
            the message to send
        """
        if not self.is_paused:
            self._msg_sender(self.name, target, msg, prio)
        else:
            self._paused_messages_post.append((target, msg, prio))

    def __str__(self):
        return 'MessagePassingComputation({})'.format(self.name)


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

    """

    def __init__(self, name, comp_def: ComputationDef):
        if comp_def is None or name is None:
            raise ValueError('ComputationDef and name are mandatory for a DCOP '
                             'computation')
        super().__init__(name)
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


class VariableComputation(DcopComputation):
    """
    A VariableComputation is a dcop computation that is responsible for
    selecting the value of a variable.

    """

    def __init__(self, variable: Variable, comp_def: ComputationDef):
        if variable is None:
            raise ValueError('Variable object is mandatory for a '
                             'VariableComputation')
        super().__init__(variable.name, comp_def)
        self._variable = variable
        self.__value__ = None  # NEVER access this directly
        self.__cost__ = None  # NEVER access this directly
        self._previous_val = None

    @property
    def variable(self)-> Variable:
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

    def value_selection(self, val, cost):
        """
        When the computation selects a value, it MUST be done by calling
        this method. This is necessary to be able to automatically monitor
        value changes.
        :param val:
        :param cost:
        """
        if val != self._previous_val:
            self.logger.info('Selecting new value: %s, %s (previous: %s, %s)',
                             val, cost, self._previous_val, self.__cost__ )
            self._on_value_selection(val, cost, self.cycle_count)
            self._previous_val = val
            self.__value__ = val
        self.__cost__ = cost


    def _on_value_selection(self, val, cost, cycle_count):
        pass


def build_computation(comp_def: ComputationDef) -> MessagePassingComputation:
    """
    Build a concrete computation instance from a computation definition.

    :param comp_def: the computation definition
    :return: a concrete MessagePassingComputation
    """
    algo_module = import_module('pydcop.algorithms.{}'
                                .format(comp_def.algo.algo))
    computation = algo_module.build_computation(comp_def)
    return computation
