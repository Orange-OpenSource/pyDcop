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
GDBA Algorithm
--------------


"""

import functools
import logging
import operator
import random
from collections import defaultdict

from typing import Iterable, Dict, Any, Tuple

from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.infrastructure.computations import Message, VariableComputation, register
from pydcop.computations_graph.constraints_hypergraph import VariableComputationNode
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import (
    RelationProtocol,
    NAryMatrixRelation,
    generate_assignment_as_dict,
    filter_assignment_dict,
    optimal_cost_value)

__author__ = "Pierre Nagellen, Pierre Rust"

GRAPH_TYPE = "constraints_hypergraph"

HEADER_SIZE = 100
UNIT_SIZE = 5


def build_computation(comp_def: ComputationDef):
    return GdbaComputation(
        comp_def.node.variable,
        comp_def.node.constraints,
        mode=comp_def.algo.mode,
        **comp_def.algo.params,
        comp_def=comp_def
    )


def computation_memory(computation: VariableComputationNode) -> float:
    """Return the memory footprint of a DBA computation.

    Notes
    -----
    With DBA, a computation must only remember the current value for each
    of it's neighbors.

    Parameters
    ----------
    computation: VariableComputationNode
        a computation in the hyper-graph computation graph

    Returns
    -------
    float:
        the memory footprint of the computation.

    """
    neighbors = set(
        (n for l in computation.neighbors for n in l.nodes if n not in computation.name)
    )
    return len(neighbors) * UNIT_SIZE


def communication_load(src: VariableComputationNode, target: str) -> float:
    """Return the communication load between two variables.

    Notes
    -----
    The main messages in DBA are the 'ok?' and 'improve' messages, which at
    most contains a value and a possible improvement. The size of the message
    does not depends on the source nor target variable, nor on their
    respective domains.

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

    return 2 * UNIT_SIZE + HEADER_SIZE


# ############################   MESSAGES   ################################
class GdbaOkMessage(Message):
    def __init__(self, value):
        super().__init__("gdba_ok", None)
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return 1

    def __str__(self):
        return "GdbaOkMessage({})".format(self.value)

    def __repr__(self):
        return "GdbaOkMessage({})".format(self.value)

    def __eq__(self, other):
        if type(other) != GdbaOkMessage:
            return False
        if self.value == other.value:
            return True
        return False


class GdbaImproveMessage(Message):
    def __init__(self, improve):
        super().__init__("gdba_improve", None)
        self._improve = improve

    @property
    def improve(self):
        return self._improve

    @property
    def size(self):
        return 1

    def __str__(self):
        return "GdbaImproveMessage({})".format(self.improve)

    def __repr__(self):
        return "GdbaImproveMessage({})".format(self.improve)

    def __eq__(self, other):
        if type(other) != GdbaImproveMessage:
            return False
        if self.improve == other.improve:
            return True
        return False


algo_params = [
    AlgoParameterDef("modifier", "str", ["A", "M"], "A"),
    AlgoParameterDef("violation", "str", ["NZ", "NM", "MX"], "NZ"),
    AlgoParameterDef("increase_mode", "str", ["E", "R", "C", "T"], "E"),
]


# ###########################   COMPUTATION   ############################
class GdbaComputation(VariableComputation):
    """
    GdbaComputation implements an extension of DBA to suit DCOPs. Several
    adaptations are possible. There are 3 dimensions which have several
    modes, for a total of 24 variants. They are listed below:
        EffCost: how to concretely increase the costs of constraints:
        increase of 1 or costs.
            Possible values: 'A' (Additive) or 'M' (Multiplicative)
        IsViolated: How to define that a constraint is violated.
            Possible values: 'NZ' (Non-zero cost) or 'NM' (Non-minimum) or
            'MX' (Maximum)
        IncreaseMode: Determine which costs have to be increased. Possible
        values: 'E' (single-entry) or 'C' (column) or 'R' (row) or 'T' (
        Transversal)

    See the following article for more details on those adaptation modes:
    'Distributed Breakout Algorithm: Beyond Satisfaction' (S. Okamoto,
    R. Zivan, A. Nahon, 2016)

    """

    def __init__(
        self,
        variable: Variable,
        constraints: Iterable[RelationProtocol],
        mode="min",
        modifier="A",
        violation="NZ",
        increase_mode="E",
        msg_sender=None,
        comp_def=None,
    ):
        """
        :param variable: a variable object for which this computation is
        responsible
        :param constraints: the list of constraints involving this variable
        :param modifier: The manner to modify costs. 'A' (resp. 'M') for
        additive (resp. multiplicative) manner. Defaults to 'A'
        :param violation: The criteria to determine a constraint violation.
        Defaults to 'NZ'
        :param increase_mode: The increase mode of a constraint cost
        describes which modifiers should be increased.
        Defaults to 'E'
        """

        super().__init__(variable, comp_def)

        self._msg_sender = msg_sender

        # Handling messages arriving during wrong mode
        self.__postponed_improve_messages__ = []
        self.__postponed_ok_messages__ = []

        self._waiting_mode = "starting"
        self._mode = mode
        self._modifier_mode = modifier
        self._violation_mode = violation
        self._increase_mode = increase_mode
        base_modifier = 0 if self._modifier_mode == "A" else 1
        self.__constraints__ = list()
        self.__constraints_modifiers__ = dict()
        # Transform the constraints in matrices, with also the min and max
        # values recorded
        for c in constraints:
            if type(c) != NAryMatrixRelation:
                rel_mat = NAryMatrixRelation.from_func_relation(c)
                c_array = rel_mat._m.flat
                maxi = c_array[0]
                mini = c_array[0]
                for i in c_array:
                    if i > maxi:
                        maxi = i
                    if i < mini:
                        mini = i
                rel = (rel_mat, mini, maxi)
            else:
                c_array = c._m.flat
                maxi = c_array[0]
                mini = c_array[0]
                for i in c_array:
                    if i > maxi:
                        maxi = i
                    if i < mini:
                        mini = i
                rel = (c, mini, maxi)
            self.__constraints__.append(rel)
            # The modifiers for constraints. It is a Dictionary of dictionary
            # (of dictionary ... regarding the arity of each constraint). It
            # represents the value of the modifier for each constraint asgt.
            self.__constraints_modifiers__[rel[0]] = defaultdict(lambda: base_modifier)

        self._violated_constraints = []
        # some constraints might be unary, and our variable can have several
        # constraints involving the same variable
        self._neighbors = set(
            [v for c in constraints for v in c.dimensions if v != variable]
        )
        # Agent view of its neighbors resp. for ok and improve modes
        self._neighbors_values = {}
        self._neighbors_improvements = {}
        self._my_improve = 0  # Possible improvement the agent can realize
        self._new_value = None

    @property
    def constraints(self):
        return self.__constraints__

    # WARNING: This does not return the _neighbors attribute, but only the
    # list of names of the neighbors
    @property
    def neighbors(self):
        return self._neighbors

    def on_start(self):
        # Select an initial value.
        if not self.neighbors:
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
            if self.variable.initial_value is None:
                self.value_selection(random.choice(self.variable.domain), self.current_cost)
                self.logger.info(
                    "%s gdba starts: randomly select value %s and " "send to neighbors",
                    self.variable.name,
                    self.current_value,
                )
            else:
                self.value_selection(self.variable.initial_value, self.current_cost)
                self.logger.info(
                    "%s gdba starts: select initial value %s and send to neighbors",
                    self.variable.name,
                    self.current_value,
                )
            self._send_current_value()
            self._go_to_wait_ok_mode()

    @register("gdba_ok")
    def _on_ok_msg(self, variable_name, recv_msg, t):
        self.logger.debug("%s received %s from %s", self.name, recv_msg, variable_name)
        if self._waiting_mode == "ok":
            self._handle_ok_message(variable_name, recv_msg)
        else:
            # A value message can be received during the improve mode (due to
            # async.). In this case, its handling is postponed until the next
            # iteration of wait_ok_mode
            self.logger.debug(
                "%s postponed processing of %s from %s",
                self.name,
                recv_msg,
                variable_name,
            )
            self.__postponed_ok_messages__.append((variable_name, recv_msg))

    def _handle_ok_message(self, variable_name, recv_msg):

        self._neighbors_values[variable_name] = recv_msg.value
        self.logger.debug(
            "%s processes %s from %s", self.variable.name, recv_msg, variable_name
        )
        # if we have a value for all neighbors, compute our best value for
        # conflict reduction
        if len(self._neighbors_values) == len(self.neighbors):
            self.logger.info(
                "%s received values from all neighbors : %s",
                self.name,
                self._neighbors_values,
            )
            self.__cost__, self._violated_constraints = self.compute_eval_value(
                self.current_value
            )
            # Set cost at the first step
            # Compute and send best improvement to neighbors
            bests, best_eval = self._compute_best_improvement()
            self._my_improve = self.__cost__ - best_eval
            if (self._my_improve > 0 and self._mode == "min") or (
                self._my_improve < 0 and self._mode == "max"
            ):
                self._new_value = random.choice(bests)
            else:
                self._new_value = self.current_value
            self._send_improve()
            self._go_to_wait_improve_mode()
        else:
            # Still waiting for other neighbors
            self.logger.debug(
                "%s waiting for OK values from other neighbors (got %s)",
                self.name,
                [n for n in self._neighbors_values],
            )

    def _send_improve(self):
        msg = GdbaImproveMessage(self._my_improve)
        for n in self.neighbors:
            self.post_msg(n.name, msg)
            self.logger.debug("%s has sent %s to %s", self.name, msg, n.name)

    def _compute_best_improvement(self):

        """
        Compute the best possible improvement for the current assignment.

        :return: (list of values achieving best improvement, best improvement)
        """
        best_vals = list()
        best_eval = None
        for v in self.variable.domain:
            curr_eval = self.compute_eval_value(v)[0]
            if best_eval is None:
                best_eval = curr_eval
                best_vals = [v]
            elif (self._mode == "min" and curr_eval < best_eval) or (
                self._mode == "max" and curr_eval > best_eval
            ):
                best_eval = curr_eval
                best_vals = [v]
            elif curr_eval == best_eval:
                best_vals.append(v)

        return best_vals, best_eval

    def _send_current_value(self):
        self.new_cycle()
        #       #########TO DO#########
        # This is where to put an eventual stop condition
        for n in self.neighbors:
            msg = GdbaOkMessage(self.current_value)
            self.post_msg(n.name, msg)
            self.logger.debug("%s has sent %s to %s", self.name, msg, n.name)

    def compute_eval_value(self, val):
        """
        This function computes the effective cost of the current assignment
        for the agent's variable.

        :param: a value for the variable of this object (it must be a value
        from the definition domain).

        :return: the evaluation value for the given value and the list
        of indices of the violated constraints for this value
        """
        new_eval_value = 0
        violated_constraints = []
        vars_with_cost = set()
        for c in self.__constraints__:
            (rel_mat, _, _) = c
            for v in rel_mat.dimensions:
                if hasattr(v, "cost_for_val"):
                    if v.name != self.name:
                        vars_with_cost.update([(v, self._neighbors_values[v.name])])
                    else:
                        vars_with_cost.update([(v, self.current_value)])
            if self._is_violated(c, val):
                violated_constraints.append(rel_mat)
            new_eval_value += self._eff_cost(rel_mat, val)

            vars_cost = functools.reduce(
                operator.add,
                [v.cost_for_val(v_val) for (v, v_val) in vars_with_cost],
                0,
            )
            new_eval_value += vars_cost

        return new_eval_value, violated_constraints

    def _go_to_wait_improve_mode(self):
        """
        Set _mode attribute to 'improve' and process postponed improve messages
        (if any)
        """
        self._waiting_mode = "improve"
        self.logger.debug("%s enters improve mode", self.name)
        # if improve messages were received during wiat_ok_mode, they should be
        # processed now
        for sender, msg in self.__postponed_improve_messages__:
            self.logger.debug(
                "%s processes postponed improve message %s", self.name, msg
            )
            self._handle_improve_message(sender, msg)
        self.__postponed_improve_messages__.clear()

    @register("gdba_improve")
    def _on_improve_message(self, variable_name, recv_msg, t):
        self.logger.debug("%s received %s from %s", self.name, recv_msg, variable_name)
        if self._waiting_mode == "improve":
            self._handle_improve_message(variable_name, recv_msg)
        else:
            self.logger.debug(
                "%s postpones processing of %s from %s",
                self.name,
                recv_msg,
                variable_name,
            )
            self.__postponed_improve_messages__.append((variable_name, recv_msg))

    def _handle_improve_message(self, variable_name, recv_msg):

        self._neighbors_improvements[variable_name] = recv_msg

        self.logger.debug("%s computes %s from %s", self.name, recv_msg, variable_name)

        # if messages received from all neighbors
        if len(self._neighbors_improvements) == len(self.neighbors):
            self.logger.info(
                "%s improvement messages from all neighbors: %s",
                self.name,
                self._neighbors_values,
            )
            maxi = self._my_improve
            max_list = [self.name]
            for n, msg in self._neighbors_improvements.items():
                if msg.improve > maxi:
                    maxi = msg.improve
                    max_list = [n]
                elif msg.improve == maxi:
                    max_list.append(n)

            if (self._my_improve > 0 and self._mode == "min") or (
                (self._my_improve < 0) and (self._mode == "max")
            ):
                winner = break_ties(max_list)
                if winner == self.name:  # covers all cases with self is in
                    # max_list
                    self.value_selection(
                        self._new_value, self.current_cost + self._my_improve
                    )
            elif maxi == 0:  # No neighbor can improve
                for c in self._violated_constraints:
                    self._increase_cost(c)

            # End of a cycle: clear agent view
            self._neighbors_improvements.clear()
            self._neighbors_values.clear()
            self._violated_constraints.clear()

            self._send_current_value()
            self._go_to_wait_ok_mode()
        else:
            # Still waiting for other neighbors
            self.logger.debug(
                "%s waiting for improve values from other " "neighbors (got %s)",
                self.name,
                [n for n in self._neighbors_improvements],
            )

    def _go_to_wait_ok_mode(self):
        self._waiting_mode = "ok"
        self.logger.debug("%s enters values mode", self.name)
        for sender, msg in self.__postponed_ok_messages__:
            self.logger.debug("%s processes postponed value message %s", self.name, msg)
            self._handle_ok_message(sender, msg)

        self.__postponed_ok_messages__.clear()

    def _is_violated(self, rel: Tuple[NAryMatrixRelation, float, float], val) -> bool:
        """
        Determine if a constraint is violated according to the chosen violation
        mode.
        :param rel: A tuple (NAryMatrixRelation, min_val of the matrix,
        max_val of the matrix)
        :param val: the value of the agent variable to evaluate the violation
        :return: True (resp. False) if the constraint is (rep. not) violated
        """
        m, min_val, max_val = rel
        # Keep only the assignment of variables present in the constraint
        global_asgt = self._neighbors_values.copy()
        global_asgt[self.name] = val
        tmp_assignment = filter_assignment_dict(global_asgt, m.dimensions)

        if self._violation_mode == "NZ":
            return m.get_value_for_assignment(tmp_assignment) != 0
        elif self._violation_mode == "NM":
            return m.get_value_for_assignment(tmp_assignment) != min_val
        else:  # self._violation_mode == 'MX'
            return m.get_value_for_assignment(tmp_assignment) == max_val

    def _eff_cost(self, rel: NAryMatrixRelation, val) -> float:
        """
        Compute the effective cost of a constraint with combining its base
        cost with the associated modifier value (i.e. the weight of the
        constraint at the current step)
        :param rel: a constraint given as NAryMatrixRelation.
        :param val: the value of the agent's variablefor which to compute the
        _eff_cost
        :return: the effective cost of the constraint for the current
        assignment.
        """
        # Keep only the variables present in the relation rel
        global_asgt = self._neighbors_values.copy()
        global_asgt[self.name] = val
        asgt = filter_assignment_dict(global_asgt, rel.dimensions)

        c = rel.get_value_for_assignment(asgt)
        modifier = self._get_modifier_for_assignment(rel, asgt)
        if self._modifier_mode == "A":
            c += modifier
        else:  # modifier_mode == 'M'
            c *= modifier

        return c

    def _get_modifier_for_assignment(
        self, constraint: NAryMatrixRelation, asgt: Dict[str, Any]
    ):
        """
        Search in the modifiers dictionary, the modifier corresponding to the
        given constraint and assignment, and return its value
        :param constraint: a constraint as NAryMatrixRelation
        :param asgt: a complete assignment for the constraint as a dictionary
        {variable: value}
        :return: the value of the modifier of the constraint for the given
        assignment
        """
        modifier = self.__constraints_modifiers__[constraint]
        key = frozenset(asgt.items())

        return modifier[key]

    def _increase_modifier(self, constraint: NAryMatrixRelation, asgt: Dict[str, Any]):
        """
        Increase the modifier corresponding to the arguments
        :param constraint: a constraint as NAryMatrixRelation
        :param asgt: a complete assignment for the constraint
        """
        modifier = self.__constraints_modifiers__[constraint]
        key = frozenset(asgt.items())

        modifier[key] += 1

    def _increase_cost(self, constraint: NAryMatrixRelation):
        """
        Increase the cost(s) of a constraint according to the given
        increase_mode
        :param constraint: a constraint as NAryMatrixRelation
        :return:
        """
        asgt = self._neighbors_values.copy()
        asgt[self.name] = self.current_value
        self.logger.debug("%s increases cost for %s", self.name, constraint)
        if self._increase_mode == "E":
            self._increase_modifier(constraint, asgt)
        elif self._increase_mode == "R":
            for val in self.variable.domain:
                asgt[self.name] = val
                self._increase_modifier(constraint, asgt)
        elif self._increase_mode == "C":
            # Creates all the assignments for the constraints, with the
            # agent variable set to its current value
            asgts = generate_assignment_as_dict(list(self._neighbors))
            for ass in asgts:
                ass[self.name] = self.current_value
                self._increase_modifier(constraint, ass)
        elif self._increase_mode == "T":
            # Creates all the assignments for the constraints
            asgts = generate_assignment_as_dict(constraint.dimensions)
            for ass in asgts:
                self._increase_modifier(constraint, ass)


def break_ties(val_list):
    return sorted(val_list)[0]
