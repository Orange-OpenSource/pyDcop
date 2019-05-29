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

MaxSum: Belief-propagation DCOP algorithm
-----------------------------------------

Implementation of the MaxSum algorithm

We try to make as few assumption on the way the algorithm is run,
and especially on the distribution of variables and factor on agents.
In particular, we do not force here a factor and a variable to belong to
the same agent and thus variables and factors are implemented completely
independently.
To run the Algorithm, factor and variable must be distributed on agents (
who will 'run' them).



"""


import logging

from collections import defaultdict

from pydcop.dcop.objects import VariableNoisyCostFunc, Variable
from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.algorithms import maxsum
from pydcop.dcop.relations import generate_assignment_as_dict
from pydcop.infrastructure.computations import (
    DcopComputation,
    VariableComputation,
    register,
)

# Avoid using symbolic infinity as it is currently not correctly
# (de)serialized
# INFINITY = float('inf')
INFINITY = 100000

STABILITY_COEFF = 0.1

HEADER_SIZE = 0
UNIT_SIZE = 1


SAME_COUNT = 4

# constants for memory costs and capacity
FACTOR_UNIT_SIZE = 1
VARIABLE_UNIT_SIZE = 1

GRAPH_TYPE = "factor_graph"
logger = logging.getLogger("pydcop.maxsum")


def build_computation(comp_def: ComputationDef):
    if comp_def.node.type == "VariableComputation":
        logger.debug(f"Building variable computation {comp_def}")
        return MaxSumVariableComputation(comp_def=comp_def)
    if comp_def.node.type == "FactorComputation":
        logger.debug(f"Building factor computation {comp_def}")
        return MaxSumFactorComputation(comp_def=comp_def)


# MaxSum and AMaxSum have the same definitions for communication load
# and computation footprints.
computation_memory = maxsum.computation_memory
communication_load = maxsum.communication_load

algo_params = [
    AlgoParameterDef("infinity", "int", None, 10000),
    AlgoParameterDef("stability", "float", None, 0.1),
    AlgoParameterDef("damping", "float", None, 0.0),
    AlgoParameterDef("stability", "float", None, STABILITY_COEFF),
]





class MaxSumFactorComputation(DcopComputation):
    """
    FactorAlgo encapsulate the algorithm running at factor's node.

    """

    def __init__(self, comp_def=None):
        assert comp_def.algo.algo == "amaxsum"
        super().__init__(comp_def.node.factor.name, comp_def)
        self.mode = comp_def.algo.mode
        self.factor = comp_def.node.factor

        # costs : messages for our variables, used to store the content of the
        # messages received from our variables.
        # v -> d -> costs
        # For each variable, we keep a dict mapping the values for this
        # variable to an associated cost.
        self._costs = {}


        # A dict var_name -> (message, count)
        self._prev_messages = defaultdict(lambda: (None, 0))

        self._valid_assignments_cache = None
        self._valid_assignments()

    @property
    def variables(self):
        """
        :return: The list of variables objects the factor depends on.
        """
        return self.factor.dimensions

    def footprint(self):
        return computation_memory(self.computation_def.node)

    def on_start(self):
        # Only unary factors (leaf in the graph) needs to send their costs at
        # init.Each leaf factor sends his costs to its only variable.
        # When possible it is better to use a variable with integrated costs
        # instead of a variable with an unary relation representing costs.
        if len(self.variables) == 1:
            for v in self.variables:
                costs_v = maxsum.factor_costs_for_var(self.factor, v, self._costs, self.mode)
                self.post_msg(v.name, maxsum.MaxSumMessage(costs_v))
                self.logger.info(
                    f"Sending init messages from factor {self.name} -> {v.name} : {costs_v}"
                )

    @register("max_sum")
    def _on_maxsum_msg(self, var_name, msg, t):
        """
        Handling messages from variables nodes.

        :param var_name: name of the variable node that sent this messages
        :param msg: the cost sent by the variable var_name
        a d -> cost table, where
          * d is a value from the domain of var_name
          * cost is the sum of the costs received from all other factors
            except f for this value d for the domain.
        """
        self._costs[var_name] = msg.costs

        # Wait until we received costs from all our variables before sending
        # our own costs
        if len(self._costs) == len(self.factor.dimensions):
            stable = True
            for v in self.variables:
                if v.name != var_name:
                    costs_v = maxsum.factor_costs_for_var(self.factor, v, self._costs, self.mode)
                    same, same_count = self._match_previous(v.name, costs_v)
                    if not same or same_count < SAME_COUNT:
                        self.logger.debug(
                            f"Sending from factor {self.name} -> {v.name} : {costs_v}")
                        self.post_msg(v.name, maxsum.MaxSumMessage(costs_v))
                        self._prev_messages[v.name] = costs_v, same_count + 1
                    else:
                        self.logger.debug(
                            f"Not sending (same) from factor {self.name} -> {v.name} : {costs_v}")

        else:
            self.logger.debug(
                f" Still waiting for costs from all  the variables {self._costs.keys()}"
            )

    def _valid_assignments(self):
        """
        Populates a cache with all valid assignments for the factor
        managed by the algorithm.

        :return: a list of all assignments returning a non-infinite value
        """
        # Fixme: extract as a function
        # FIXME: does not take into account min / max
        if self._valid_assignments_cache is None:
            self._valid_assignments_cache = []
            all_vars = self.factor.dimensions[:]
            for assignment in generate_assignment_as_dict(all_vars):
                if self.factor(**assignment) != INFINITY:
                    self._valid_assignments_cache.append(assignment)
        return self._valid_assignments_cache

    def _match_previous(self, v_name, costs):
        """
        Check if a cost message for a variable v_name match the previous 
        message sent to that variable.

        :param v_name: variable name
        :param costs: costs sent to this factor
        :return:
        """
        prev_costs, count = self._prev_messages[v_name]
        if prev_costs is not None:
            same = maxsum.approx_match(costs, prev_costs)
            return same, count
        else:
            return False, 0


class MaxSumVariableComputation(VariableComputation):
    """
    Maxsum Computation for variable.

    Parameters
    ----------
    comp_def: ComputationDef
    """
    def __init__(self,comp_def: ComputationDef = None):
        """

        :param variable: variable object
        :param factor_names: a list containing the names of the factors that
        depend on the variable managed by this algorithm
        :param msg_sender: the object that will be used to send messages to
        neighbors, it must have a  post_msg(sender, target_name, name) method.
        """
        super().__init__(comp_def.node.variable, comp_def)

        assert comp_def.algo.algo == "amaxsum"
        assert (comp_def.algo.mode == "min") or (comp_def.algo.mode == "max")

        self.mode = comp_def.algo.mode

        # Add noise to the variable, on top of cost if needed
        # TODO: make this configurable through parameters
        self._variable = VariableNoisyCostFunc(
            self.variable.name,
            self.variable.domain,
            cost_func=lambda x: self.variable.cost_for_val(x),
            initial_value=self.variable.initial_value,
        )

        # The list of factors (names) this variables is linked with
        self._factors = [link.factor_node for link in comp_def.node.links]

        # costs : this dict is used to store, for each value of the domain,
        # the associated cost sent by each factor this variable is involved
        # with. { factor : {domain value : cost }}
        self._costs = {}

        self._prev_messages = defaultdict(lambda: (None, 0))

        self.damping = comp_def.algo.params["damping"]
        self.logger.info("Running maxsum with damping %s", self.damping)

    def on_start(self) -> None:
        """
        Startup handler for MaxSum variable computations.

        At startup, a variable select an initial value and send its cost to the factors
        it depends on.
        """

        # select our initial value
        if self.variable.initial_value:
            self.value_selection(self.variable.initial_value, None)
        else:
            self.value_selection(*maxsum.select_value(self.variable, self._costs, self.mode))
        self.logger.info(f"Initial value selected {self.current_value}")

        # Send our costs to the factors we depends on.
        for f in self._factors:
            costs_f = maxsum.costs_for_factor(self.variable, f, self._factors, self._costs)
            self.logger.info(
                f"Sending init msg from variable {self.name} to factor {f} : {costs_f}"
            )
            self.post_msg(f, maxsum.MaxSumMessage(costs_f))

    @register("max_sum")
    def _on_maxsum_msg(self, factor_name, msg, t):
        """
        Handling cost message from a neighbor factor.

        Parameters
        ----------
        factor_name: str
            the name of that factor that sent us this message.
        msg: MaxSumMessage
            a message whose content is a map { d -> cost } where:
            * d is a value from the domain of this variable
            * cost if the minimum cost of the factor when taking value d
        """
        self._costs[factor_name] = msg.costs

        # select our value
        self.value_selection(*maxsum.select_value(self.variable, self._costs, self.mode))

        # Compute and send our own costs to all other factors.
        # If our variable has his own costs, we must sent them back even
        # to the factor which sent us this message, as integrated costs are
        # similar to an unary factor and with an unary factor we would have
        # sent these costs back to the original sender:
        # factor -> variable -> unary_cost_factor -> variable -> factor
        fs = self._factors.copy()
        fs.remove(factor_name)

        for f_name in fs:
            costs_f = maxsum.costs_for_factor(self.variable, f_name, self._factors, self._costs)

            same, same_count = self._match_previous(f_name, costs_f)
            if not same or same_count < SAME_COUNT:
                self.logger.debug(f"Sending from variable {self.name} -> {f_name} : {costs_f}")
                self.post_msg(f_name, maxsum.MaxSumMessage(costs_f))
                self._prev_messages[f_name] = costs_f, same_count + 1

            else:
                self.logger.debug(f"Not sending (similar) from {self.name} -> {f_name} : {costs_f}")

    def _match_previous(self, f_name, costs):
        """
        Check if a cost message for a factor f_name match the previous message
        sent to that factor.

        :param f_name: factor name
        :param costs: costs sent to this factor
        :return:
        """
        prev_costs, count = self._prev_messages[f_name]
        if prev_costs is not None:
            same = maxsum.approx_match(costs, prev_costs)
            return same, count
        else:
            return False, 0


