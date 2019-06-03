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

Max-Sum :cite:`farinelli_decentralised_2008` is an incomplete inference-based DCOP
algorithm.

This is a **synchronous implementation** of Max-Sum, where messages are sent and received
in cycles. For an asynchronous implementation,
see. :ref:`A-Max-Sum<implementation_reference_algorithms_amaxsum>`


Algorithm Parameters
^^^^^^^^^^^^^^^^^^^^

FIXME: stop_cycle
FIXME: damping
FIXME: stability
FIXME: infinity


Example
^^^^^^^

::

    pydcop solve -algo maxsum  \\
      --algo_param stop_cycle:30 \\
     -d adhoc graph_coloring_csp.yaml

FIXME: add results

See Also
^^^^^^^^
:ref:`A-Max-Sum<implementation_reference_algorithms_amaxsum>`: an asynchronous implementation of
Max-Sum.


"""
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from collections import defaultdict


from pydcop.algorithms import ComputationDef, AlgoParameterDef
from pydcop.computations_graph.factor_graph import (
    FactorComputationNode,
    VariableComputationNode,
)
from pydcop.dcop.objects import Variable, VariableNoisyCostFunc
from pydcop.dcop.relations import Constraint, generate_assignment_as_dict
from pydcop.infrastructure.computations import (
    DcopComputation,
    SynchronousComputationMixin,
    VariableComputation,
    register,
    Message,
)

GRAPH_TYPE = "factor_graph"
logger = logging.getLogger("pydcop.maxsum")

# Avoid using symbolic infinity as it is currently not correctly
# (de)serialized
# INFINITY = float('inf')
INFINITY = 100000

SAME_COUNT = 4

STABILITY_COEFF = 0.1

HEADER_SIZE = 0
UNIT_SIZE = 1

# constants for memory costs and capacity
FACTOR_UNIT_SIZE = 1
VARIABLE_UNIT_SIZE = 1


def build_computation(comp_def: ComputationDef):
    if comp_def.node.type == "VariableComputation":
        logger.debug(f"Building variable computation {comp_def}")
        return MaxSumVariableComputation(comp_def=comp_def)
    if comp_def.node.type == "FactorComputation":
        logger.debug(f"Building factor computation {comp_def}")
        return MaxSumFactorComputation(comp_def=comp_def)


def computation_memory(
    computation: Union[FactorComputationNode, VariableComputationNode]
) -> float:
    """Memory footprint associated with the maxsum computation node.

    Notes
    -----
    Two formulations of the memory footprint are possible for factors :
    * If the constraint is given by a function (intentional), the factor
      only needs to keep the costs sent by each variable and the footprint
      is the total size of these cost vectors.
    * If the constraints is given extensively the size of the hypercube of
      costs must also be accounted for.

    Parameters
    ----------
    computation: FactorComputationNode or VariableComputationNode
        A computation node for a factor or a variable in the factor-graph.

    Returns
    -------
    float:
        the memory footprint of the computation.
    """
    if isinstance(computation, FactorComputationNode):
        # Memory footprint associated with the factor computation f.
        # For Maxsum, it depends on the size of the domain of the neighbor
        # variables.
        m = 0
        for v in computation.variables:
            domain_size = len(v.domain)
            m += domain_size * FACTOR_UNIT_SIZE
        return m

    elif isinstance(computation, VariableComputationNode):
        # For Maxsum, the memory footprint a variable computations depends
        #  on the number of  neighbors in the factor graph.
        domain_size = len(computation.variable.domain)
        num_neighbors = len(list(computation.links))
        return num_neighbors * domain_size * VARIABLE_UNIT_SIZE

    raise ValueError(
        "Invalid computation node type {}, maxsum only defines "
        "VariableComputationNodeand FactorComputationNode".format(computation)
    )


def communication_load(
    src: Union[FactorComputationNode, VariableComputationNode], target: str
) -> float:
    """The communication cost of an edge between a variable and a factor.

    Parameters
    ----------
    src: VariableComputationNode
        The ComputationNode for the source variable.
    target: str
        the name of the other variable `src` is sending messages to

    Return
    ------
    float:
        the size of messages between computation and target.
    """
    if isinstance(src, VariableComputationNode):
        d_size = len(src.variable.domain)
        return UNIT_SIZE * d_size + HEADER_SIZE

    elif isinstance(src, FactorComputationNode):
        for v in src.variables:
            if v.name == target:
                d_size = len(v.domain)
                return UNIT_SIZE * d_size + HEADER_SIZE
        raise ValueError(
            "Could not find variable {} in constraint of factor "
            "{}".format(target, src)
        )

    raise ValueError(
        "maxsum communication_load only supports "
        "VariableComputationNode and FactorComputationNode, "
        "invalid computation: " + str(src)
    )


class MaxSumMessage(Message):
    def __init__(self, costs: Dict):
        super().__init__("max_sum", None)
        self._costs = costs

    @property
    def costs(self):
        return self._costs

    @property
    def size(self):
        # Max sum messages are dictionaries from values to costs:
        return len(self._costs) * 2

    def __str__(self):
        return "MaxSumMessage({})".format(self._costs)

    def __repr__(self):
        return "MaxSumMessage({})".format(self._costs)

    def __eq__(self, other):
        if type(other) != MaxSumMessage:
            return False
        if self.costs == other.costs:
            return True
        return False

    def _simple_repr(self):
        r = {"__module__": self.__module__, "__qualname__": self.__class__.__qualname__}

        # When building the simple repr when transform the dict into a pair
        # of list to avoid problem when serializing / de-serializing the repr.
        # The costs dic often contains int as key, when converting to an from
        # json (which only support string for keys in dict), we would
        # otherwise loose the type information and restore the dict with str
        # keys.
        vals, costs = zip(*self._costs.items())
        r["vals"] = vals
        r["costs"] = costs
        return r

    @classmethod
    def _from_repr(cls, r):
        vals = r["vals"]
        costs = r["costs"]

        return MaxSumMessage(dict(zip(vals, costs)))


# Some semantic type definition, to make things easier to read and check:
VarName = str
FactorName = str
VarVal = Any
Cost = float


class MaxSumFactorComputation(SynchronousComputationMixin, DcopComputation):
    def __init__(self, comp_def: ComputationDef):
        assert comp_def.algo.algo == "maxsum"
        super().__init__(comp_def.node.factor.name, comp_def)
        self.logger.warning(f"Neiborghs {self.neighbors}")

        self.mode = comp_def.algo.mode
        self.factor = comp_def.node.factor
        self.variables = self.factor.dimensions

        # costs : messages for our variables, used to store the content of the
        # messages received from our variables.
        # {v -> {d -> costs} }
        # For each variable, we keep a dict mapping the values for this
        # variable to an associated cost.
        self._costs: Dict[VarName, Dict[VarVal:Cost]] = {}

        # A dict var_name -> (message, count)
        self._prev_messages = defaultdict(lambda: (None, 0))

    def on_start(self):

        # Only unary factors (leaf in the graph) needs to send their costs at
        # init.Each leaf factor sends his costs to its only variable.
        # When possible it is better to use a variable with integrated costs
        # instead of a variable with an unary relation representing costs.
        if len(self.variables) == 1:
            self.logger.info(f"Sending init costs of unary factor {self.name}")
            msg_debug = []
            for v in self.variables:
                costs_v = self._costs_for_var(v)
                self.post_msg(v.name, MaxSumMessage(costs_v))
                msg_debug.append((v.name, costs_v))

            if self.logger.isEnabledFor(logging.DEBUG):
                debug = f"Unary factor : init msg {self.name} \n"
                for dest, msg in msg_debug:
                    debug += f"  * {self.name} -> {dest} : {msg}\n"
                self.logger.debug(debug + "\n")
            else:
                self.logger.info(
                    f"Init messages for {self.name} to {[c for c, _ in msg_debug]}"
                )

    @register("max_sum")
    def on_msg(self, variable_name, recv_msg, t):
        # No implementation here, simply used to declare the kind of message supported
        # by this computation
        pass

    def on_new_cycle(self, messages, cycle_id) -> Optional[List]:

        for sender, (message, t) in messages.items():
            self._costs[sender] = message.costs

        for v in self.variables:
            costs_v = factor_costs_for_var(self.factor, v, self._costs, self.mode)
            same, same_count = self._match_previous(v.name, costs_v)
            if not same or same_count < SAME_COUNT:
                self.logger.debug(
                    f"Sending from factor {self.name} -> {v.name} : {costs_v}"
                )
                self.post_msg(v.name, MaxSumMessage(costs_v))
                self._prev_messages[v.name] = costs_v, same_count + 1
            else:
                self.logger.debug(
                    f"Not sending (same) from factor {self.name} -> {v.name} : {costs_v}"
                )

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
            same = approx_match(costs, prev_costs)
            return same, count
        else:
            return False, 0

def factor_costs_for_var(factor: Constraint, variable: Variable, recv_costs, mode: str):
    """
    Computes the marginals to be send by a factor to a variable


    The content of this message is a table d -> mincost where
    * d is a value of the domain of the variable v
    * mincost is the minimum value of f when the variable v take the
      value d

    :param variable: the variable we want to send the costs to
    :return: a mapping { value => cost}
    where value is all the values from the domain of 'variable'
    costs is the cost when 'variable'  == 'value'

    Parameters
    ----------
    factor: Constraint
        the factor that will send these cost to `variable`
    variable: Variable

    recv_costs: Dict
        a dict containing the costs received from other variables
    mode: str
        "min" or "max"

    Returns
    -------
    Dict:
        a dict that associates a cost to each value in the domain of `variable`

    """
    # TODO: support passing list of valid assignment as param
    costs = {}
    other_vars = factor.dimensions[:]
    other_vars.remove(variable)
    for d in variable.domain:
        # for each value d in the domain of v, calculate min cost (a)
        # where a is any assignment where v = d
        # cost (a) = f(a) + sum( costvar())
        # where costvar is the cost received from our other variables

        mode_opt = INFINITY if mode == "min" else -INFINITY
        optimal_value = mode_opt

        for assignment in generate_assignment_as_dict(other_vars):
            assignment[variable.name] = d
            f_val = factor(**assignment)
            if f_val == INFINITY:
                continue

            sum_cost = 0
            # sum of the costs from all other variables
            for another_var, var_value in assignment.items():
                if another_var == variable.name:
                    continue
                if another_var in recv_costs:
                    if var_value not in recv_costs[another_var]:
                        # If there is no cost for this value, it means it
                        #  is infinite (as infinite cost are not included
                        # in messages) and we can stop adding costs.
                        sum_cost = mode_opt
                        break
                    sum_cost += recv_costs[another_var][var_value]
                else:
                    # we have not received yet costs from variable v
                    pass

            current_val = f_val + sum_cost
            if (optimal_value > current_val and mode == "min") or (
                optimal_value < current_val and mode == "max"
            ):

                optimal_value = current_val

        if optimal_value != mode_opt:
            costs[d] = optimal_value

    return costs


class MaxSumVariableComputation(SynchronousComputationMixin, VariableComputation):
    def __init__(self, comp_def: ComputationDef):
        super().__init__(comp_def.node.variable, comp_def)
        assert comp_def.algo.algo == "maxsum"
        self.mode = comp_def.algo.mode
        self.logger.warning(f"Neiborghs {self.neighbors}")

        # The list of factors (names) this variables is linked with
        self.factors = [link.factor_node for link in comp_def.node.links]
        # costs : this dict is used to store, for each value of the domain,
        # the associated cost sent by each factor this variable is involved
        # with. { factor : {domain value : cost }}
        self.costs = {}

        # to store previous messages, necessary to detect convergence
        self._prev_messages = defaultdict(lambda: (None, 0))

        # TODO: restore support for damping
        # self.damping = comp_def.algo.params["damping"]
        # self.logger.info("Running maxsum with damping %s", self.damping)

        # Add noise to the variable, on top of cost if needed
        # TODO: make this configurable through parameters
        self._variable = VariableNoisyCostFunc(
            self.variable.name,
            self.variable.domain,
            cost_func=lambda x: self.variable.cost_for_val(x),
            initial_value=self.variable.initial_value,
        )

    @register("max_sum")
    def on_msg(self, variable_name, recv_msg, t):
        # No implementation here, simply used to declare the kind of message supported
        # by this computation
        pass

    def on_start(self) -> None:
        # Select our initial value
        if self.variable.initial_value is not None:
            self.value_selection(self.variable.initial_value)
        else:
            self.value_selection(*select_value(self.variable, self.costs, self.mode))
        self.logger.info(f"Initial value selected {self.current_value}")

        # Send our costs to the factors we depends on.
        for f in self.factors:
            costs_f = costs_for_factor(self.variable, f, self.factors, self.costs)
            self.logger.info(
                f"Sending init msg from variable {self.name} to factor {f} : {costs_f}"
            )
            self.post_msg(f, MaxSumMessage(costs_f))

    def on_new_cycle(self, messages, cycle_id) -> Optional[List]:

        for sender, (message, t) in messages.items():
            self.costs[sender] = message.costs

        # select our value, based on new costs
        self.value_selection(*select_value(self.variable, self.costs, self.mode))

        # Compute and send our own costs to  factors.

        for f_name in self.factors:
            costs_f = costs_for_factor(
                self.variable, f_name, self.factors, self.costs
            )

            same, same_count = self._match_previous(f_name, costs_f)
            if not same or same_count < SAME_COUNT:
                self.logger.debug(
                    f"Sending from variable {self.name} -> {f_name} : {costs_f}"
                )
                self.post_msg(f_name, MaxSumMessage(costs_f))
                self._prev_messages[f_name] = costs_f, same_count + 1

            else:
                self.logger.debug(
                    f"Not sending (similar) from {self.name} -> {f_name} : {costs_f}"
                )

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
            same = approx_match(costs, prev_costs)
            return same, count
        else:
            return False, 0


def select_value(variable: Variable, costs: Dict, mode: str) -> Tuple[Any, float]:
    """
    select the value for `variable` with the best cost / reward (depending on `mode`)

    Returns
    -------
    a Tuple containing the selected value and the corresponding cost for
    this computation.
    """

    # If we have received costs from all our factor, we can select a
    # value from our domain.
    d_costs = {d: variable.cost_for_val(d) for d in variable.domain}
    for d in variable.domain:
        for f_costs in costs.values():
            if d not in f_costs:
                # As infinite costs are not included in messages,
                # if there is not cost for this value it means the costs
                # is infinite and we can stop adding other costs.
                d_costs[d] = INFINITY if mode == "min" else -INFINITY
                break
            d_costs[d] += f_costs[d]

    from operator import itemgetter

    if mode == "min":
        optimal_d = min(d_costs.items(), key=itemgetter(1))
    else:
        optimal_d = max(d_costs.items(), key=itemgetter(1))

    return optimal_d[0], optimal_d[1]


def costs_for_factor(
    variable: Variable, factor: FactorName, factors: List[Constraint], costs: Dict
) -> Dict[VarVal, Cost]:
    """
    Produce the message that must be sent to factor f.

    The content if this message is a d -> cost table, where
    * d is a value from the domain
    * cost is the sum of the costs received from all other factors except f
    for this value d for the domain.

    Parameters
    ----------
    variable: Variable
        the variable sending the message
    factor: str
        the name of the factor the message will be sent to
    factors: list of Constraints
        the constraints this variables depends on
    costs: dict
        the accumulated costs received by the variable from all factors

    Returns
    -------
    Dict:
        a dict containing a cost for each value in the domain of the variable
    """
    # If our variable has integrated costs, add them
    msg_costs = {d: variable.cost_for_val(d) for d in variable.domain}

    sum_cost = 0
    for d in variable.domain:
        for f in [f for f in factors if f != factor and f in costs]:
            f_costs = costs[f]
            if d not in f_costs:
                msg_costs[d] = INFINITY
                break
            c = f_costs[d]
            sum_cost += c
            msg_costs[d] += c

    # Experimentally, when we do not normalize costs the algorithm takes
    # more cycles to stabilize
    # return {d: c for d, c in msg_costs.items() if c != INFINITY}

    # Normalize costs with the average cost, to avoid exploding costs
    avg_cost = sum_cost / len(msg_costs)
    normalized_msg_costs = {
        d: c - avg_cost for d, c in msg_costs.items() if c != INFINITY
    }
    msg_costs = normalized_msg_costs

    # FIXME: restore damping support
    # prev_costs, count = self._prev_messages[factor]
    # damped_costs = {}
    # if prev_costs is not None:
    #     for d, c in msg_costs.items():
    #         damped_costs[d] = self.damping * prev_costs[d] + (1 - self.damping) * c
    #     self.logger.warning("damping : replace %s with %s", msg_costs, damped_costs)
    #     msg_costs = damped_costs

    return msg_costs


def approx_match(costs, prev_costs):
    """
    Check if a cost message match the previous message.

    Costs are considered to match if the variation is bellow STABILITY_COEFF.

    :param costs: costs as a dict val -> cost
    :param prev_costs: previous costs as a dict val -> cost
    :return: True if the cost match
    """

    for d, c in costs.items():
        prev_c = prev_costs[d]
        if prev_c != c:
            delta = abs(prev_c - c)
            if prev_c + c != 0:
                if not ((2 * delta / abs(prev_c + c)) < STABILITY_COEFF):
                    return False
            else:
                return False
    return True
