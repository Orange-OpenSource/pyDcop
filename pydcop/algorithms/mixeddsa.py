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
import operator
import random

import functools
from typing import Dict, List, Tuple

from pydcop.dcop.relations import RelationProtocol, generate_assignment_as_dict, \
    filter_assignment_dict

from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.infrastructure.computations import Message, VariableComputation, \
    register

from pydcop.computations_graph.constraints_hypergraph import \
    VariableComputationNode

__author__ = "Pierre Nagellen, Pierre Rust"

GRAPH_TYPE = 'constraints_hypergraph'

INFINITY = float("inf")

HEADER_SIZE = 100
UNIT_SIZE = 5


def build_computation(comp_def: ComputationDef):
    return MixedDsaComputation(comp_def.node.variable,
                               comp_def.node.constraints,
                               mode=comp_def.algo.mode,
                               **comp_def.algo.params,
                               comp_def=comp_def)


def computation_memory(computation: VariableComputationNode) -> float :
    """Return the memory footprint of a DSA computation.

    Notes
    -----
    With DSA, a computation must only remember the current value for each
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
    The only message in DSA is the 'value' messages, which simply contains
    the current value.

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
    return  UNIT_SIZE + HEADER_SIZE


algo_params = [
    AlgoParameterDef('proba_hard', 'float', None, 0.7),
    AlgoParameterDef('proba_soft', 'float', None, 0.5),
    AlgoParameterDef('variant', 'str', ['A', 'B', 'C'], 'B'),
    AlgoParameterDef('stop_cycle', 'int', None, 0),
]


class MixedDsaMessage(Message):
    def __init__(self, value):
        super().__init__('mixed_dsa_value', None)
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return 1

    def __str__(self):
        return 'MixedDsaMessage({})'.format(self.value)

    def __repr__(self):
        return 'MixedDsaMessage({})'.format(self.value)

    def __eq__(self, other):
        if type(other) != MixedDsaMessage:
            return False
        if self.value == other.value:
            return True
        return False


class MixedDsaComputation(VariableComputation):
    """
    MixedDsaComputation implements several variant of th DSA. It is said to be
    mixed because it is designed to help improving DSA performances in DCOPs
    with both hard and soft constraints.

    See. the following article for a complete description of DSA:
    'Distributed stochastic search and distributed breakout: properties,
    comparison and applications to constraint optimization problems in sensor
    networks', Zhang Weixiong & al, 2005

    """

    def __init__(self, variable, constraints, variant='B', proba_hard=0.7,
                 proba_soft=0.7, mode='min', comp_def=None):
        """

        :param variable a variable object for which this computation is
        responsible
        :param constraints: the list of constraints involving this variable
        :param variant: the variant of the DSA algorithm : 'A' for DSA-A,
        etc.. possible values avec 'A', 'B' and 'C'
        :param proba_hard : the probability to change the value in case of
        the number of violated hard constraints can be decreased
        :param proba_soft : the probability to change the value in case the
        cost on hard constraints can't be improved, but the cost on soft
        constraints can
        :param mode: optimization mode, 'min' for minimization and 'max' for
        maximization. Defaults to 'min'.

        """
        super().__init__(variable, comp_def)

        self.proba_hard = proba_hard
        self.proba_soft = proba_soft
        self.variant = variant
        self.mode = mode
        # some constraints might be unary, and our variable can have several
        # constraints involving the same variable
        self._neighbors = set([v.name for c in constraints
                               for v in c.dimensions if v != variable])
        self._neighbors_values = {}
        self._postponed_messages = list()

        self.hard_constraints = list()
        self.soft_constraints = list()
        self._violated_hard_cons = list()
        self._curr_dcop_cost = None
        self.__optimum_dict__ = {}
        # We do not use pydcop.dcop.relations.find_optimum() to distinguish
        # hard and soft constraints
        for c in constraints:
            hard = False
            variables = [v for v in c.dimensions if v != self._variable]
            boundary = None
            for asgt in generate_assignment_as_dict(variables):
                rel = c.slice(filter_assignment_dict(asgt, c.dimensions))
                for val in self._variable.domain:
                    rel_val = rel(val)
                    if boundary is None:
                        boundary = rel_val
                    elif self.mode == 'max' and rel_val > boundary:
                        boundary = rel_val
                    elif self.mode == 'min' and rel_val < boundary:
                        boundary = rel_val
                    if rel_val == float("inf") or rel_val == -float("inf"):
                        hard = True
            self.__optimum_dict__[c.name] = boundary
            if hard:
                self.hard_constraints.append(c)
            else:
                self.soft_constraints.append(c)

        if not self.hard_constraints:
            global INFINITY
            INFINITY = float("inf")

    def on_start(self):
        if self.variable.initial_value is None:
            self.value_selection(random.choice(self.variable.domain),
                                 self.current_cost)
            self.logger.debug(
                '%s dsa starts (%s %s %s) : randomly select value %s and '
                'send to neighbors', self.variable.name,
                self.variant, self.proba_hard, self.proba_soft,
                self.current_value)
        else:
            self.value_selection(self.variable.initial_value,
                                 self.current_cost)
            self.logger.debug(
                '%s dsa starts (%s %s %s) : select fixed initial value %s and '
                'send to neighbors', self.variable.name,
                self.variant, self.proba_hard, self.proba_soft,
                self.current_value)

        # send a value msg to all neighbors
        self._send_value()

        # as everything is asynchronous, we might have received our
        # neighbors values even before starting this algorithm. In this case
        # we must treat them now.
        self._on_neighbors_values()

    @register("mixed_dsa_value")
    def _on_value_msg(self, variable_name, recv_msg, t):
        if variable_name not in self._neighbors_values:
            self._neighbors_values[variable_name] = recv_msg.value
            self.logger.debug('%s received value %s from %s',
                              self.variable.name,
                              recv_msg.value, variable_name)
        else:  # The message is too early
            self.logger.debug('%s received the value of %s for the next turn. '
                              'Processing of this message is postponed until'
                              ' next turn', self.name, variable_name)
            self._postponed_messages.append((variable_name, recv_msg.value))

        self._on_neighbors_values()

    def _on_neighbors_values(self):
        # if we have a value for all neighbors, compute our best value for
        # conflict reduction
        # We also check that we have already selected an initial value,
        # otherwise it makes no sense to compute our gain.
        if len(self._neighbors_values) == len(self._neighbors) and \
                        self.current_value is not None:
            self.logger.debug('%s received values from all neighbors : %s',
                              self.variable.name,
                              self._neighbors_values)
            nb_violated_cons, dcop_cost, bests = self._compute_best_value()
            # Compute the cost before computing the gain
            current_asgt = self._neighbors_values.copy()
            current_asgt[self.name] = self.current_value
            self._curr_dcop_cost, self._violated_hard_cons = \
                self._compute_dcop_cost(current_asgt)

            delta_dcsp, delta_dcop = (len(self._violated_hard_cons) -
                                      nb_violated_cons,
                                      self._curr_dcop_cost - dcop_cost)

            self.logger.debug('%s best value : %s with %s violated '
                              'hard constraints and a cost %s for other '
                              'constraints - improvement %s, %s ',
                              self.variable.name, bests, nb_violated_cons,
                              dcop_cost, delta_dcsp, delta_dcop)
            eff_cost = self._eff_cost(dcop_cost, nb_violated_cons)

            if delta_dcsp > 0:
                if self.proba_hard > random.random():
                    self.value_selection(random.choice(bests), eff_cost)
                    self.logger.info('%s select new value %s with cost %s',
                                     self.variable.name, self.current_value,
                                     self.current_cost)
            elif delta_dcsp == 0:
                if (self.mode == 'min' and delta_dcop > 0) or \
                        (self.mode == 'max' and delta_dcop < 0):
                    if self.proba_soft > random.random():
                        self.value_selection(random.choice(bests), eff_cost)
                        self.logger.info('%s select new value %s with cost %s',
                                         self.variable.name, self.current_value,
                                         self.current_cost)
                    else:
                        self.logger.info('%s has potential improvement but not'
                                         'value change', self.name)
                elif delta_dcop == 0:
                    # MixDSA-B and C may still change their value when no
                    # improvement is possible, if there are still conflicts.
                    # This helps escaping local optima
                    if nb_violated_cons > 0:
                        if len(bests) > 1 and self.proba_hard > random.random():
                            bests.remove(self.current_value)
                            self.value_selection(random.choice(bests), eff_cost)
                        self.logger.info('%s select new value %s with same cost'
                                         ' (%s, %s) (MixDSA B/C)',
                                         self.variable.name,
                                         self.current_value,
                                         nb_violated_cons, dcop_cost)
                    elif self.exists_violated_soft_constraint() and\
                                    self.variant in ['B', 'C']:
                        if len(bests) > 1 and self.proba_soft > random.random():
                            bests.remove(self.current_value)
                            self.value_selection(random.choice(bests), eff_cost)

                        self.logger.info('%s select new value %s with same cost'
                                         ' (%s, %s) (MixDSA B/C)',
                                         self.variable.name,
                                         self.current_value,
                                         nb_violated_cons, dcop_cost)

                elif delta_dcop == 0 and self.variant == 'C':
                    # MixDSA-C may change the value event with no conflict nor
                    # improvement.
                    if len(bests) > 1 and\
                            min(self.proba_hard, self.proba_soft) > \
                                    random.random():
                        bests.remove(self.current_value)
                        self.value_selection(random.choice(bests), eff_cost)
                        self.logger.info(
                            '%s select new value %s with no conflict '
                            'and same cost  (%s, %s) (DSA-C)',
                            self.variable.name, self.current_value,
                            nb_violated_cons, dcop_cost)
                else:
                    self.logger.debug(
                        '%s has no possible improvement : do not change '
                        'value', self.name)

            self._neighbors_values.clear()
            self._send_value()
            # Beginning of next turn: process the postponed messages
            while self._postponed_messages:
                neighbor, value = self._postponed_messages.pop()
                self.logger.debug(
                    '%s process postponed received value %s from %s',
                    self.name, value, neighbor)
                self._neighbors_values[neighbor] = value

            # In case every neighbor messages have already been received
            # and postponed
            if self._neighbors:
                self._on_neighbors_values()

        else:
            # Still waiting for other neighbors
            self.logger.debug(
                '%s waiting for values from other neighbors (got %s)',
                self.name, [n for n in self._neighbors_values])

    def _compute_best_value(self):
        asgt = self._neighbors_values.copy()
        best_dcop = None
        best_dcsp = len(self.hard_constraints) + 1
        best_vals = list()
        for val in self.variable.domain:
            asgt[self.name] = val
            cost, violated = self._compute_dcop_cost(asgt)
            nb_violated = len(violated)
            if nb_violated < best_dcsp:
                best_dcop = cost
                best_dcsp = nb_violated
                best_vals = [val]
            elif nb_violated == best_dcsp:
                if (cost < best_dcop and self.mode == 'min') or \
                        (cost > best_dcop and self.mode == 'max'):
                    best_dcop = cost
                    best_vals = [val]
                elif cost == best_dcop:
                    best_vals.append(val)

        return best_dcsp, best_dcop, best_vals

    def _send_value(self):
        self.new_cycle()
        for n in self._neighbors:
            msg = MixedDsaMessage(self.current_value)
            self.post_msg(n, msg)

    def _compute_dcop_cost(self, assignment, soft_cons=None, hard_cons=None) \
            -> Tuple[float, List[RelationProtocol]]:
        """
        Compute the cost for the given assignment, and the list of violated
        hard constraints. The cost computed does not include the infinite
        costs of the violated constraints.
        :param assignment: A full assignement for the relation
        :param soft_cons: a list of soft constraints. Default to the
        computation's soft constraints
        :param hard_cons: a list of hard constraints. Default to the
        computation's hard constraints
        :return: a couple: dcop_cost, list of violated constraints
        """
        softs = self.soft_constraints if soft_cons is None else soft_cons
        hards = self.hard_constraints if hard_cons is None else hard_cons
        # Cost for constraints:
        cost = functools.reduce(operator.add, [f(**filter_assignment_dict(
            assignment, f.dimensions)) for f in softs], 0)
        # Cost for variable, if any:
        concerned_vars = set(v for c in softs for v in c.dimensions)
        concerned_vars.update(v for c in hards for v in c.dimensions)
        for v in concerned_vars:
            if hasattr(v, 'cost_for_val'):
                cost += v.cost_for_val(assignment[v.name])

        hard_violated = list()
        for f in hards:
            c_cost = f(**filter_assignment_dict(assignment, f.dimensions))
            if c_cost == INFINITY:
                hard_violated.append(f)
                # We do not set the cost to infinity yet, so that we can
                # favorize solutions with the lowest cost (without counting
                # the violated hard constraints)
            else:
                cost += c_cost

        return cost, hard_violated

    def _eff_cost(self, dcop_cost: float, nb_violated_constraints: int) -> \
            float:
        """
        Compute the effective cost of a relation for a given assignment from
        the DCOP cost and a number of violated constraints.

        :param dcop_cost: a float representing a cost
        :param nb_violated_constraints: an int, number of violated hard
        constraints
        :return: Symbolic infinity if 'violated_constraints' not empty,
        'dcop_cost' otherwise.
        """
        if nb_violated_constraints:
            return INFINITY
        return dcop_cost

    def exists_violated_soft_constraint(self) -> bool:
        """
        Tells if there is a violated soft constraint regarding the current
        assignment
        :return: a boolean
        """
        for c in self.soft_constraints:
            asgt = self._neighbors_values.copy()
            asgt[self.name] = self.current_value
            const = c(filter_assignment_dict(asgt, c.dimensions))
            if const != self.__optimum_dict__[c.name]:
                return True
        return False
