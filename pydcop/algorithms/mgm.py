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
import random
import operator

import functools
from typing import Any
from typing import Dict
from typing import Iterable, Set

from pydcop.algorithms import filter_assignment_dict, find_arg_optimal, \
    ComputationDef
from pydcop.infrastructure.computations import Message, VariableComputation

from pydcop.computations_graph.constraints_hypergraph import ConstraintLink, \
    VariableComputationNode
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import RelationProtocol

GRAPH_TYPE = 'constraints_hypergraph'

HEADER_SIZE = 100
UNIT_SIZE = 5
BREAK_MODES = ['lexic', 'random']


def algo_name() -> str:
    """

    Returns
    -------
    The name of the algorithm implemented by this module : 'mgm'
    """
    return __name__.split('.')[-1]


def build_computation(comp_def: ComputationDef):
    return MgmComputation(comp_def.node.variable, comp_def.node.constraints,
                          mode=comp_def.algo.mode,
                          **comp_def.algo.params,
                          comp_def=comp_def)


def computation_memory(computation: VariableComputationNode) -> float:
    """Return the memory footprint of a MGM computation.

    Notes
    -----
    With MGM, a computation must only remember the current value for each
    of it's neighbors.

    Parameters
    ----------
    computation: VariableComputationNode
        a computation in the hyper-graph computation graph
    links: iterable of links
        links for this computation node. links maps to constraints in the
        computation graph and can be hyper-edges (when the arity of the
        constraint is > 2)

    Returns
    -------
    float:
        the memory footprint of the computation.

    """
    neighbors = set((n for l in computation.links for n in l.nodes
                     if n not in computation.name))
    return len(neighbors) * UNIT_SIZE


def communication_load(src: VariableComputationNode, target: str) -> float:
    """Return the communication load between two variables.

    Notes
    -----
    The main messages in MGM are the 'value' and 'gain' messages, which both
    contains a simple value.

    Parameters
    ----------
    src: VariableComputationNode
        The ComputationNode for the source variable.
    target: str
        the name of the other variable `src` is sending messages to

    Returns
    -------
    float
        The size of messages sent from the src variable to the target variable.
    """
    return UNIT_SIZE + HEADER_SIZE


# ############################   MESSAGES   ################################
class MgmValueMessage(Message):
    """
    A class to send Value messages to neighbors to inform them of the
    variable value

    """

    def __init__(self, value):
        super().__init__('mgm_value', None)
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return 1

    def __str__(self):
        return 'MgmValueMessage({})'.format(self.value)

    def __repr__(self):
        return 'MgmValueMessage({})'.format(self.value)

    def __eq__(self, other):
        if type(other) != MgmValueMessage:
            return False
        if self.value == other.value:
            return True
        return False


# Basically the same class than MgmValueMessage, but we need two classes to
# differentiate the kind of messages received for postponing processing when
# not in the good state
class MgmGainMessage(Message):
    """
    A class designed to send Gain messages to inform neighbors about the
    maximum (local) gain the variable can achieve if it changes its value
    """

    def __init__(self, value, random_nb=0):
        super().__init__('mgm_gain', None)
        self._value = value
        self._random_nb = random_nb

    @property
    def value(self):
        return self._value

    @property
    def random_nb(self):
        return self._random_nb

    @property
    def size(self):
        return 1

    def __str__(self):
        return 'MgmGainMessage({})'.format(self.value)

    def __repr__(self):
        return 'MgmGainMessage({})'.format(self.value)

    def __eq__(self, other):
        if type(other) != MgmGainMessage:
            return False
        if self.value == other.value:
            return True
        return False


def algo_params(params: Dict[str, str]):
    """
    DSA support two parameters:

    * the value used as 'infinity', returned as the cost of a violated
    constraint (it must map the value used in your dcop definition)

    * 'max_path' : an upper bound for the maximum distance between two
    agents in the graph. Ideally you could use the graph diameter or simply
    the number of variables in the problem. It is use for termination
    detection (which in DBA only works is there is a solution to the problem).

    :param params: a dict containing name and values for parameters

    :return: a Dict with all dsa paremeters (either their default value or
    the values extracted form `params`
    """
    mgm_params = {
        'break_mode': 'lexic',
    }
    if 'break_mode' in params:
        if (params['break_mode'] == 'lexic') or \
                (params['break_mode'] == 'random'):
            mgm_params['break_mode'] = params['break_mode']
        else:
            raise ValueError("'break_mode' parameter for MGM must be in {}"
                             .format(BREAK_MODES))
    remaining_params = set(params) - {'break_mode'}
    if remaining_params:
        raise ValueError('Unknown parameter(s) for MGM : {}'
                         .format(remaining_params))
    return mgm_params


# ###########################   COMPUTATION   ############################
class MgmComputation(VariableComputation):
    """
    MgmComputation implements MGM algorithm as described in 'Distributed
    Algorithms for DCOP: A Graphical-Game-Base Approach' (R. Maheswaran,
    J. Pearce, M. Tambe, 2004).

    """

    def __init__(self, variable: Variable,
                 utilities: Iterable[RelationProtocol],
                 mode='min', msg_sender=None, logger=None,
                 break_mode='lexic',
                 comp_def=None):
        """
        :param variable: a variable object for which this computation is
                         responsible
        :param utilities: the list of utilities/constraints involving this
                          variable
        :param mode: optimization mode, 'min' or 'max'. Defaults to min
        """

        super().__init__(variable, comp_def)
        self._msg_handlers['mgm_value'] = self._on_value_msg
        self._msg_handlers['mgm_gain'] = self._on_gain_msg

        self._msg_sender = msg_sender
        self.logger = logger if logger is not None \
            else logging.getLogger('pydcop.algo.mgm.' + variable.name)

        self.__utilities__ = list(utilities)
        self._mode = mode  # min or max

        # Handling messages arriving during wrong mode
        self.__postponed_gain_messages__ = []
        self.__postponed_value_messages__ = []
        # _state is set to 'values' or 'gains', according what the agent is
        # currently waiting for
        self._state = 'starting'

        # Some constraints might be unary, and our variable can have several
        # constraints involving the same variable
        self._neighbors = set([v.name for c in utilities
                               for v in c.dimensions if v != variable])

        # Agent view of its neighbors resp. for values and gains state
        self._neighbors_values = {}  # type: Dict[str, Any]
        self._neighbors_gains = {}  # type: Dict[str, MgmGainMessage]
        self._gain = None
        self._new_value = None

        self.__break_mode__ = break_mode
        self.__random_nb__ = 0  # used in case break_mode is 'random'

    def footprint(self):
        return computation_memory(self.computation_def.node)

    @property
    def random_nb(self) -> float:
        return self.__random_nb__

    @property
    def utilities(self) -> Iterable[RelationProtocol]:
        return self.__utilities__

    @property
    def neighbors(self) -> Set[str]:
        return self._neighbors

    def on_start(self):
        """
        Start the algorithm (and select an initial random value for the
        variable if none is set by default) when the agent is started

        """
        # randomly select a value
        if self.variable.initial_value is None:
            self.value_selection(random.choice(self.variable.domain), None)
            self.logger.info('%s mgm starts: randomly select value %s and '
                             'send to neighbors',
                             self.variable.name, self.current_value)
        else:
            self.value_selection(self.variable.initial_value, None)
            self.logger.info('%s mgm starts: select initial value %s and '
                             'send to neighbors',
                             self.variable.name, self.current_value)
        self._wait_for_values()

    def _on_value_msg(self, variable_name, recv_msg, t):
        """
        Postpones (resp. launches) the processing of the received Value
        message, if the state is set to 'gains' (resp. 'values')
        :param variable_name: name of the sender
        :param recv_msg: received Value message (MgmValueMessage)

        """
        if self._state == 'values':
            self.logger.debug('%s received variable value %s from %s '
                              'and processes it', self.variable.name,
                              recv_msg.value, variable_name)
            self._handle_value_message(variable_name, recv_msg)
        else:
            self.logger.debug('%s received variable value %s from %s and '
                              'postponed  its processing',
                              self.variable.name, recv_msg.value,
                              variable_name)
            self.__postponed_value_messages__.append((variable_name, recv_msg))

    def _handle_value_message(self, variable_name, recv_msg):
        """
        Processes a received Value message to determine what is the best gain
        the varaible can achieve

        :param variable_name: name of the sender
        :param recv_msg: A MgmValueMessage

        """
        self._neighbors_values[variable_name] = recv_msg.value
        # if we have a value for all neighbors, compute the best value for
        # conflict reduction
        if len(self._neighbors_values) == len(self._neighbors):
            self.logger.debug('%s received values from all neighbors : %s',
                              self.name, self._neighbors_values)
            # Compute the current_cost on the first step (initialization) of
            # the algorithm
            if self.current_cost is None:
                reduced_cs = []
                concerned_vars = set()
                cost = 0
                for c in self.utilities:
                    asgt = filter_assignment_dict(self._neighbors_values,
                                                  c.dimensions)
                    reduced_cs.append(c.slice(asgt))
                    cost = functools.reduce(
                        operator.add, [f(self.current_value) for f in
                                       reduced_cs])
                    # Cost for variable, if any:
                    concerned_vars.update(c.dimensions)

                for v in concerned_vars:
                    if hasattr(v, 'cost_for_val'):
                        if v.name == self.name:
                            cost += v.cost_for_val(self.current_value)
                        else:
                            cost += v.cost_for_val(
                                self._neighbors_values[v.name])

                self.value_selection(self.current_value, cost)

            new_values, val_cost = self._compute_best_value()
            self._gain = self.current_cost - val_cost
            if ((self._mode == 'min') & (self._gain > 0)) or \
                    ((self._mode == 'max') & (self._gain < 0)):
                self._new_value = random.choice(new_values)
            else:
                self._new_value = self.current_value

            self.logger.info('Best local value for %s: %s %s (neighbors: %s)',
                             self.name, self._new_value, self._gain,
                             self._neighbors_values)

            self._send_gain()

            self._wait_for_gains()
        else:
            # Still waiting for other neighbors
            self.logger.debug('%s waiting for values from other neighbors: %s',
                              self.name,
                              [n for n in self._neighbors
                               if n not in self._neighbors_values])

    def _send_gain(self):
        """
        Send an MgmGainMessage to inform neighbors of the best possible gain
        the variable can realize.

        """
        self.__random__ = random.random()
        msg = MgmGainMessage(self._gain, self.__random__)
        self.logger.debug('%s sends gain message %s to %s', self.name, msg,
                          self.neighbors)
        for n in self.neighbors:
            self.post_msg(n, msg)

    def _send_value(self):
        """
        Send an MgmValueMessage to inform neighbors of the currant value of
        the variable.

        """
        self.new_cycle()
        msg = MgmValueMessage(self.current_value)
        self.logger.debug('%s sends value message %s to %s', self.name, msg,
                          self.neighbors)
        for n in self._neighbors:
            self.post_msg(n, msg)

    def _wait_for_gains(self):
        """
        Change variable state to 'gains' and compute postponed value messages
        if any.

        """
        self._state = 'gain'
        self.logger.debug('%s enters gain state', self.name)
        for msg in self.__postponed_gain_messages__:
            self.logger.debug('%s processes postponed message %s', self.name,
                              msg)
            self._handle_gain_message(msg[0], msg[1])
        self.__postponed_gain_messages__.clear()

    def _compute_best_value(self):
        """
        Compute the best evaluation the variable can have wrt the current
        values of neighbors.

        :return: (list of values the variable that lead to the best
        evaluation, best evaluation)

        """
        reduced_cs = []
        concerned_vars = set()

        for c in self.utilities:
            asgt = filter_assignment_dict(self._neighbors_values, c.dimensions)
            reduced_cs.append(c.slice(asgt))
            concerned_vars.update(c.dimensions)
        var_val, rel_val = find_arg_optimal(
            self.variable,
            lambda x: functools.reduce(operator.add, [f(x) for f in
                                                      reduced_cs]),
            self._mode)
        # Add the cost for each variable value if any
        for var in concerned_vars:
            if hasattr(var, 'cost_for_val'):
                if var.name == self.name:
                    rel_val += var.cost_for_val(self.current_value)
                else:
                    rel_val += var.cost_for_val(
                        self._neighbors_values[var.name])

        return var_val, rel_val

    # #############################GAIN STATE##################################
    def _on_gain_msg(self, variable_name, recv_msg, t):
        """
        Postpones (resp. launches) the processing of the received Gain
        message, if the state is set to 'values' (resp. 'gains')
        :param variable_name: name of the sender
        :param recv_msg: received Gain message (MgmGainMessage)

        """

        if self._state == 'gain':
            self.logger.debug('%s received gain %s from %s and processes it',
                              self.variable.name, recv_msg.value,
                              variable_name)
            self._handle_gain_message(variable_name, recv_msg)
        else:
            self.logger.debug('%s received gain %s from %s and postponed its'
                              ' processing', self.variable.name,
                              recv_msg.value, variable_name)
            self.__postponed_gain_messages__.append((variable_name, recv_msg))

    def _handle_gain_message(self, variable_name, recv_msg):
        """
        Processes a received Gain message to determine if the variable is
        allowed to change its value.

        :param variable_name: name of the sender
        :param recv_msg: A MgmGainMessage

        """

        self._neighbors_gains[variable_name] = (recv_msg.value,
                                                recv_msg.random_nb)

        # if messages received from all neighbors
        if len(self._neighbors_gains) == len(self._neighbors):
            gains = {var: gain for var, gain in self._neighbors_gains.items()}
            self.logger.debug('%s has all gains : %s -  %s', self.name,
                              self._gain,
                              gains)
            # determine if can change value and send ok message to neighbors
            max_neighbors = max([gain for gain, _ in gains.values()])
            if self._gain > max_neighbors:
                self.logger.info('%s selects new value "%s", best gain: %s > '
                                 '%s',
                                 self.name, self._new_value, self._gain,
                                 gains)
                self.value_selection(self._new_value,
                                     self.current_cost - self._gain)
            elif self._gain == max_neighbors:
                # same gain, break ties through variable ordering to
                # determine which variable can change its value
                self._break_ties(max_neighbors)
            else:
                self.logger.info('%s doe not change value : not the best '
                                 'gain %s < %s ', self.name, self._gain,
                                 max_neighbors)

            self._neighbors_gains.clear()
            self._neighbors_values.clear()
            self._wait_for_values()
        else:
            # Still waiting for other neighbors
            self.logger.debug('%s waiting for gain msg from other neighbors '
                              ': %s', self.name,
                              [n for n in self._neighbors
                               if n not in self._neighbors_gains])

    def _break_ties(self, max_gain):
        if self.__break_mode__ == random:
            ties = sorted([(rand_nb, name) for name, (gain, rand_nb) in
                           self._neighbors_gains.items()
                           if gain == max_gain] + [(self.random_nb,
                                                    self.name)],
                          )
            if ties[0][1] == self.name:
                self.logger.info('Won ties for equal gain %s , %s '
                                 'selects new value "%s - %s"', self._gain,
                                 self.name, self._new_value, ties)
                self.value_selection(self._new_value,
                                     self.current_cost - self._gain)
            else:
                self.logger.info('Lost ties for equal gain %s , %s does '
                                 'not change value to "%s" - %s',
                                 self._gain, self.name, self._new_value,
                                 ties)
        else:
            ties = sorted([k for k, (gain, _) in
                           self._neighbors_gains.items() if
                           gain == max_gain] + [self.name])
            if ties[0] == self.name:
                self.logger.info('Won ties for equal gain %s , %s '
                                 'selects new value "%s - %s"', self._gain,
                                 self.name, self._new_value, ties)
                self.value_selection(self._new_value,
                                     self.current_cost - self._gain)
            else:
                self.logger.info('Lost ties for equal gain %s , %s does '
                                 'not change value to "%s" - %s',
                                 self._gain, self.name, self._new_value,
                                 ties)

    def _wait_for_values(self):
        """
        Change variable state to 'values'

        """
        self.logger.debug('%s enters values state', self.name)
        # End of a cycle: clear agent view

        self._state = 'values'
        self._send_value()
        for msg in self.__postponed_value_messages__:
            self.logger.debug('%s processes postponed message %s', self.name,
                              msg)
            self._handle_value_message(msg[0], msg[1])
        self.__postponed_value_messages__.clear()

    def __str__(self):
        return 'MgmComputation({})'.format(self.name)

    def __repr__(self):
        return 'MgmComputation({}, {})'.format(
            self.name,self.__utilities__)
