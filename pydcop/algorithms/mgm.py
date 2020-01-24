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

MGM : Maximum Gain Message
--------------------------

Algorithm Parameters
^^^^^^^^^^^^^^^^^^^^
MGM supports two parameters:

* break_mode
* stop_cycle

Example
^^^^^^^

TODO

"""

import functools
import logging
import operator
import random
from typing import Any
from typing import Dict
from typing import Iterable, Set

from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.computations_graph.constraints_hypergraph import VariableComputationNode
from pydcop.dcop.relations import (
    RelationProtocol,
    filter_assignment_dict,
    find_arg_optimal,
    optimal_cost_value,
)
from pydcop.infrastructure.computations import Message, VariableComputation, register

GRAPH_TYPE = "constraints_hypergraph"

HEADER_SIZE = 100
UNIT_SIZE = 5
BREAK_MODES = ["lexic", "random"]


"""
MGM supports two paramaters: 
* break_mode
* stop_cycle
"""
algo_params = [
    AlgoParameterDef("break_mode", "str", ["lexic", "random"], "lexic"),
    AlgoParameterDef("stop_cycle", "int", None, 0),
]


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
    neighbors = set(
        (n for l in computation.links for n in l.nodes if n not in computation.name)
    )
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
        super().__init__("mgm_value", None)
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return 1

    def __str__(self):
        return "MgmValueMessage({})".format(self.value)

    def __repr__(self):
        return "MgmValueMessage({})".format(self.value)

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
        super().__init__("mgm_gain", None)
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
        return "MgmGainMessage({})".format(self.value)

    def __repr__(self):
        return "MgmGainMessage({})".format(self.value)

    def __eq__(self, other):
        if type(other) != MgmGainMessage:
            return False
        if self.value == other.value:
            return True
        return False


# ###########################   COMPUTATION   ############################
class MgmComputation(VariableComputation):
    """
    MgmComputation implements MGM algorithm as described in 'Distributed
    Algorithms for DCOP: A Graphical-Game-Base Approach' (R. Maheswaran,
    J. Pearce, M. Tambe, 2004).

    Parameters
    ----------
    computation_definition: ComputationDef


    """

    def __init__(self, computation_definition: ComputationDef):

        assert computation_definition.algo.algo == "mgm"
        assert (computation_definition.algo.mode == "min") or (
            computation_definition.algo.mode == "max"
        )

        super().__init__(computation_definition.node.variable, computation_definition)

        self.__utilities__ = list(computation_definition.node.constraints)
        self._mode = computation_definition.algo.mode  # min or max

        # Handling messages arriving during wrong mode
        self.__postponed_gain_messages__ = []
        self.__postponed_value_messages__ = []
        # _state is set to 'values' or 'gains', according what the agent is
        # currently waiting for
        self._state = "starting"

        # Some constraints might be unary, and our variable can have several
        # constraints involving the same variable
        self._neighbors = set(
            [
                v.name
                for c in self.__utilities__
                for v in c.dimensions
                if v != self.variable
            ]
        )

        # Agent view of its neighbors resp. for values and gains state
        self._neighbors_values = {}  # type: Dict[str, Any]
        self._neighbors_gains = {}  # type: Dict[str, MgmGainMessage]
        self._gain = None
        self._new_value = None

        self.stop_cycle = computation_definition.algo.param_value("stop_cycle")
        self.break_mode = computation_definition.algo.param_value("break_mode")
        self._random_nb = 0  # used in case break_mode is 'random'

    @property
    def random_nb(self) -> float:
        return self._random_nb

    @property
    def utilities(self) -> Iterable[RelationProtocol]:
        return self.__utilities__

    @property
    def neighbors(self) -> Set[str]:
        return self._neighbors

    def on_start(self):
        """
        Start the algorithm and select an initial value for the variable when the
        agent is started.
        """

        if not self._neighbors:
            # If a variable has no neighbors, we must select its final value immediately
            # as it will never receive any message.
            value, cost = optimal_cost_value(self._variable, self._mode)
            self.value_selection(value, cost)

            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"Select initial value {self.current_value} "
                    f"based on cost function for var {self._variable.name}"
                )
            self.finished()

        else:
            # The variable has neighbors: select a value, which might change
            # once we receive messages.
            if self.variable.initial_value is None:
                self.value_selection(random.choice(self.variable.domain), None)
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"Select initial random value {self.current_value}"
                    )
            else:
                self.value_selection(self.variable.initial_value, None)
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(f"Select initial value {self.current_value}")
            self._wait_for_values()

    @register("mgm_value")
    def _on_value_msg(self, variable_name, recv_msg, t):
        """
        Postpones (resp. launches) the processing of the received Value
        message, if the state is set to 'gains' (resp. 'values')
        :param variable_name: name of the sender
        :param recv_msg: received Value message (MgmValueMessage)

        """
        if self._state == "values":
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Received variable value {recv_msg.value} from {variable_name}"
                )
            self._handle_value_message(variable_name, recv_msg)
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Postponing variable value {recv_msg.value} from {variable_name}"
                )
            self.__postponed_value_messages__.append((variable_name, recv_msg))

    def _handle_value_message(self, variable_name, recv_msg):
        """
        Processes a received Value message to determine what is the best gain
        the variable can achieve

        :param variable_name: name of the sender
        :param recv_msg: A MgmValueMessage

        """
        self._neighbors_values[variable_name] = recv_msg.value
        # if we have a value for all neighbors, compute the best value for
        # conflict reduction
        if len(self._neighbors_values) == len(self._neighbors):
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Received values from all neighbors : {self._neighbors_values}"
                )
            # Compute the current_cost on the first step (initialization) of
            # the algorithm
            if self.current_cost is None:
                reduced_cs = []
                concerned_vars = set()
                cost = 0
                for c in self.utilities:
                    asgt = filter_assignment_dict(self._neighbors_values, c.dimensions)
                    reduced_cs.append(c.slice(asgt))
                    cost = functools.reduce(
                        operator.add, [f(self.current_value) for f in reduced_cs]
                    )
                    # Cost for variable, if any:
                    concerned_vars.update(c.dimensions)

                for v in concerned_vars:
                    if v.name == self.name:
                        cost += v.cost_for_val(self.current_value)
                    else:
                        cost += v.cost_for_val(self._neighbors_values[v.name])

                self.value_selection(self.current_value, cost)

            new_values, val_cost = self._compute_best_value()
            self._gain = self.current_cost - val_cost
            if ((self._mode == "min") & (self._gain > 0)) or (
                (self._mode == "max") & (self._gain < 0)
            ):
                self._new_value = random.choice(new_values)
            else:
                self._new_value = self.current_value

            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"Best local value for {self.name}: {self._new_value}"
                    f" {self._gain} (neighbors: {self._neighbors_values})"
                )

            self._send_gain()

            self._wait_for_gains()
        else:
            # Still waiting for other neighbors
            if self.logger.isEnabledFor(logging.DEBUG):
                waited = [n for n in self._neighbors if n not in self._neighbors_values]
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"Waiting for values from other neighbors: {waited}"
                    )

    def _send_gain(self):
        """
        Send an MgmGainMessage to inform neighbors of the best possible gain
        the variable can realize.

        """
        self.__random__ = random.random()
        msg = MgmGainMessage(self._gain, self.__random__)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Sends gain message {msg} to {self.neighbors}")
        for n in self.neighbors:
            self.post_msg(n, msg)

    def _send_value(self):
        """
        Send an MgmValueMessage to inform neighbors of the currant value of
        the variable.

        """
        self.new_cycle()
        if self.stop_cycle and self.cycle_count >= self.stop_cycle:
            self.finished()
            return
        msg = MgmValueMessage(self.current_value)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Sends value message {msg} to {self.neighbors}")
        for n in self._neighbors:
            self.post_msg(n, msg)

    def _wait_for_gains(self):
        """
        Change variable state to 'gains' and compute postponed value messages
        if any.

        """
        self._state = "gain"
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Enters gain state")
        for msg in self.__postponed_gain_messages__:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Processes postponed message {msg}")
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
            lambda x: functools.reduce(operator.add, [f(x) for f in reduced_cs]),
            self._mode,
        )
        # Add the cost for each variable value if any
        for var in concerned_vars:
            if var.name == self.name:
                rel_val += var.cost_for_val(self.current_value)
            else:
                rel_val += var.cost_for_val(self._neighbors_values[var.name])

        return var_val, rel_val

    # #############################GAIN STATE##################################
    @register("mgm_gain")
    def _on_gain_msg(self, variable_name, recv_msg, t):
        """
        Postpones (resp. launches) the processing of the received Gain
        message, if the state is set to 'values' (resp. 'gains')
        :param variable_name: name of the sender
        :param recv_msg: received Gain message (MgmGainMessage)

        """

        if self._state == "gain":
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Received gain {recv_msg.value} from {variable_name}"
                )
            self._handle_gain_message(variable_name, recv_msg)
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Postponing gain {recv_msg.value} from {variable_name}"
                )
            self.__postponed_gain_messages__.append((variable_name, recv_msg))

    def _handle_gain_message(self, variable_name, recv_msg):
        """
        Processes a received Gain message to determine if the variable is
        allowed to change its value.

        :param variable_name: name of the sender
        :param recv_msg: A MgmGainMessage

        """

        self._neighbors_gains[variable_name] = (recv_msg.value, recv_msg.random_nb)

        # if messages received from all neighbors
        if len(self._neighbors_gains) == len(self._neighbors):
            gains = {var: gain for var, gain in self._neighbors_gains.items()}
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Has all gains {self._gain}, {gains}")
            # determine if can change value and send ok message to neighbors
            max_neighbors = max([gain for gain, _ in gains.values()])
            if self._gain > max_neighbors:
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"Selects new value {self._new_value}, "
                        f"best gain: {self._gain} > {gains}"
                    )
                self.value_selection(self._new_value, self.current_cost - self._gain)
            elif self._gain == max_neighbors:
                # same gain, break ties through variable ordering to
                # determine which variable can change its value
                self._break_ties(max_neighbors)
            else:
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"Doe not change value : "
                        f"not the best gain {self._gain} < {max_neighbors} "
                    )

            self._neighbors_gains.clear()
            self._neighbors_values.clear()
            self._wait_for_values()
        else:
            # Still waiting for other neighbors
            if self.logger.isEnabledFor(logging.DEBUG):
                waited = [n for n in self._neighbors if n not in self._neighbors_gains]
                self.logger.debug(
                    f"Waiting for gain msg from other neighbors : {waited}"
                )

    def _break_ties(self, max_gain):
        if self.break_mode == random:
            ties = sorted(
                [
                    (rand_nb, name)
                    for name, (gain, rand_nb) in self._neighbors_gains.items()
                    if gain == max_gain
                ]
                + [(self.random_nb, self.name)]
            )
            if ties[0][1] == self.name:
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"Won random ties for equal gain {self._gain} , "
                        f"selects new value {self._new_value} - {ties}"
                    )
                self.value_selection(self._new_value, self.current_cost - self._gain)
            else:
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"Lost random ties for equal gain {self._gain} , "
                        f"does not change value to {self._new_value} - {ties}"
                    )
        else:
            ties = sorted(
                [
                    k
                    for k, (gain, _) in self._neighbors_gains.items()
                    if gain == max_gain
                ]
                + [self.name]
            )
            if ties[0] == self.name:
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"Won lexic ties for equal gain {self._gain} , "
                        f"selects new value {self._new_value} - {ties}"
                    )
                self.value_selection(self._new_value, self.current_cost - self._gain)
            else:
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"Lost lexic ties for equal gain {self._gain} , does "
                        f"not change value to {self._new_value} - {ties}"
                    )

    def _wait_for_values(self):
        """
        Change variable state to 'values'

        """
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Enters values state")
        # End of a cycle: clear agent view

        self._state = "values"
        self._send_value()
        for msg in self.__postponed_value_messages__:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Processing postponed message {msg}")

            self._handle_value_message(msg[0], msg[1])
        self.__postponed_value_messages__.clear()
