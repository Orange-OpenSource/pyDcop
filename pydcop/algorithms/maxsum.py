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

**damping**
  amount of dumping [0-1]

**damping_nodes**
  nodes that apply damping to messages: "vars", "factors", "both" or "none"

**stability**
  stability detection coefficient

**noise**
  noise level for variable

**start_messages**
  nodes that initiate messages : "leafs", "leafs_vars", "all"


FIXME: add support for stop_cycle


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


algo_params = [
    AlgoParameterDef("damping", "float", None, 0.5),
    AlgoParameterDef(
        "damping_nodes", "str", ["vars", "factors", "both", "none"], "both"
    ),
    AlgoParameterDef("stability", "float", None, STABILITY_COEFF),
    AlgoParameterDef("noise", "float", None, 0.01),
    AlgoParameterDef("start_messages", "str", ["leafs", "leafs_vars", "all"], "leafs"),
]


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

        self.damping = comp_def.algo.params["damping"]
        self.damping_nodes = comp_def.algo.params["damping_nodes"]
        self.stability_coef = comp_def.algo.params["stability"]
        self.start_messages = comp_def.algo.params["start_messages"]
        self.logger.info(f"Running maxsum with params: {comp_def.algo.params}")

        # A dict var_name -> (message, count)
        self._prev_messages = defaultdict(lambda: (None, 0))

    def on_start(self):

        # Only unary factors (leaf in the graph) needs to send their costs at
        # init.Each leaf factor sends his costs to its only variable.
        # When possible it is better to use a variable with integrated costs
        # instead of a variable with an unary relation representing costs.
        if len(self.variables) == 1 and self.start_messages in ["leafs", "leafs_vars"]:
            for v in self.variables:
                costs_v = factor_costs_for_var(
                    self.factor, v, self._costs, self.mode
                )
                self.post_msg(v.name, MaxSumMessage(costs_v))
                self.logger.info(
                    f"Sending init messages from factor {self.name} -> {v.name} : {costs_v}"
                )
        elif self.start_messages == "all":
            for v in self.variables:
                costs_v = factor_costs_for_var(
                    self.factor, v, self._costs, self.mode
                )
                self.post_msg(v.name, MaxSumMessage(costs_v))
                self.logger.info(
                    f"Sending init messages from factor {self.name} -> {v.name} : {costs_v}"
                )

    @register("max_sum")
    def on_msg(self, variable_name, recv_msg, t):
        # No implementation here, simply used to declare the kind of message supported
        # by this computation
        pass

    def footprint(self) -> float:
        return computation_memory(self.computation_def.node)

    def on_new_cycle(self, messages, cycle_id) -> Optional[List]:

        # Collect costs messages from neighbor variables for this cycle (aka iteration)
        for sender, (message, t) in messages.items():
            self._costs[sender] = message.costs

        for v in self.variables:
            costs_v = factor_costs_for_var(self.factor, v, self._costs, self.mode)
            prev_costs, count = self._prev_messages[v.name]

            # Apply damping to computed costs:
            if self.damping_nodes in ["factors", "both"]:
                costs_v = apply_damping(
                    costs_v, prev_costs, self.damping
                )

            # Check if there was enough change to send the message
            if not approx_match(
                    costs_v, prev_costs, self.stability_coef
            ):
                # Not same as previous : send
                self.logger.debug(
                    f"Sending first time from factor {self.name} -> {v.name} : {costs_v}"
                )
                self.post_msg(v.name, MaxSumMessage(costs_v))
                self._prev_messages[v.name] = costs_v, 1

            elif count < SAME_COUNT:
                # Same as previous, but not yet sent SAME_COUNT times: send
                self.logger.debug(
                    f"Sending {count} time from variable {self.name} -> {v.name} : {costs_v}"
                )
                self.post_msg(v.name, MaxSumMessage(costs_v))
                self._prev_messages[v.name] = costs_v, count + 1
            else:
                # Same and already sent SAME_COUNT times: no-send
                self.logger.debug(
                    f"Not sending (similar) from {self.name} -> {v.name} : {costs_v}"
                )

        return None


def factor_costs_for_var(factor: Constraint, variable: Variable, recv_costs, mode: str):
    """
    Computes the marginals to be send by a factor to a variable


    The content of this message is a table d -> mincost where
    * d is a value of the domain of the variable v
    * mincost is the minimum value of f when the variable v take the
      value d

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

        optimal_value = float("inf") if mode == "min" else -float("inf")

        for assignment in generate_assignment_as_dict(other_vars):
            assignment[variable.name] = d
            f_val = factor(**assignment)

            sum_cost = 0
            # sum of the costs from all other variables
            for another_var, var_value in assignment.items():
                if another_var == variable.name:
                    continue
                if another_var in recv_costs:
                    if var_value not in recv_costs[another_var]:
                        continue
                    sum_cost += recv_costs[another_var][var_value]
                else:
                    # we have not received yet costs from variable v
                    pass

            current_val = f_val + sum_cost
            if (optimal_value > current_val and mode == "min") or (
                optimal_value < current_val and mode == "max"
            ):

                optimal_value = current_val

        costs[d] = optimal_value

    return costs


class MaxSumVariableComputation(SynchronousComputationMixin, VariableComputation):
    """

    """

    def __init__(self, comp_def: ComputationDef):
        super().__init__(comp_def.node.variable, comp_def)
        assert comp_def.algo.algo == "maxsum"
        self.mode = comp_def.algo.mode
        self.damping = comp_def.algo.params["damping"]
        self.damping_nodes = comp_def.algo.params["damping_nodes"]
        self.stability_coef = comp_def.algo.params["stability"]
        self.start_messages = comp_def.algo.params["start_messages"]
        self.logger.info(f"Running maxsum with params: {comp_def.algo.params}")

        # The list of factors (names) this variables is linked with
        self.factors = [link.factor_node for link in comp_def.node.links]

        # costs : this dict is used to store, for each value of the domain,
        # the associated cost sent by each factor this variable is involved
        # with. { factor : {domain value : cost }}
        self.costs = {}

        # to store previous messages, necessary to detect convergence
        self._prev_messages = defaultdict(lambda: (None, 0))

        # Add noise to the variable, on top of cost if needed
        if comp_def.algo.params["noise"] != 0:
            self.logger.info(
                f"Adding noise on variable {comp_def.algo.params['noise']}"
            )
            self._variable = VariableNoisyCostFunc(
                self.variable.name,
                self.variable.domain,
                cost_func=lambda x: self.variable.cost_for_val(x),
                initial_value=self.variable.initial_value,
                noise_level=comp_def.algo.params["noise"],
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

        if len(self.factors) == 1 and self.start_messages == "leafs":
            # Only send costs if we are a leaf:
            single_factor = self.factors[0]
            costs_f = costs_for_factor(
                self.variable, single_factor, self.factors, self.costs
            )
            self.logger.info(
                f"Sending init msg from leaf variable {self.name} to single factor {single_factor} : {costs_f}"
            )
            self.post_msg(single_factor, MaxSumMessage(costs_f))

        elif self.start_messages in ["leafs_vars", "all"]:
            # in "leafs_vars" mode, send our costs to all the factors we depends on.
            for f in self.factors:
                costs_f = costs_for_factor(
                    self.variable, f, self.factors, self.costs
                )
                self.logger.info(
                    f"Sending init msg from variable {self.name} to factor {f} : {costs_f}"
                )
                self.post_msg(f, MaxSumMessage(costs_f))

    def on_new_cycle(self, messages, cycle_id) -> Optional[List]:

        # Collect costs messages from neighbor factors for this cycle (aka iteration)
        for sender, (message, t) in messages.items():
            self.costs[sender] = message.costs

        # select our value, based on new costs
        self.value_selection(*select_value(self.variable, self.costs, self.mode))

        # Compute and send our own costs to  factors.

        for f_name in self.factors:
            costs_f = costs_for_factor(self.variable, f_name, self.factors, self.costs)
            prev_costs, count = self._prev_messages[f_name]

            # Apply damping to computed costs:
            if self.damping_nodes in ["vars", "both"]:
                costs_f = apply_damping(costs_f, prev_costs, self.damping)

            # Check if there was enough change to send the message
            if not approx_match(costs_f, prev_costs, self.stability_coef):
                # Not same as previous : send
                self.logger.debug(
                    f"Sending first time from variable {self.name} -> {f_name} : {costs_f}"
                )
                self.post_msg(f_name, MaxSumMessage(costs_f))
                self._prev_messages[f_name] = costs_f, 1

            elif count < SAME_COUNT:
                # Same as previous, but not yet sent SAME_COUNT times: send
                self.logger.debug(
                    f"Sending {count} time from variable {self.name} -> {f_name} : {costs_f}"
                )
                self.post_msg(f_name, MaxSumMessage(costs_f))
                self._prev_messages[f_name] = costs_f, count + 1
            else:
                # Same and already sent SAME_COUNT times: no-send
                self.logger.debug(
                    f"Not sending (similar) from {self.name} -> {f_name} : {costs_f}"
                )
        return None

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
            same = approx_match(costs, prev_costs, self.stability_coef)
            return same, count
        else:
            return False, 0


def select_value(
    variable: Variable, costs: Dict[str, Dict], mode: str
) -> Tuple[Any, float]:
    """
    Select the value for `variable` with the best cost / reward (depending on `mode`)

    Parameters
    ----------
    variable: Variable
        the variable for which we need to select a value
    costs: Dict
        a dict { factorname : { value : costs}} representing the cost messages received from factors
    mode: str
        min or max
    Returns
    -------
    Tuple:
        a Tuple containing the selected value and the corresponding cost for
        this computation.
    """

    # Select a value from the domain, based on the variable cost and
    # the costs received from neighbor factors
    d_costs = {d: variable.cost_for_val(d) for d in variable.domain}
    for d in variable.domain:
        for f_costs in costs.values():
            d_costs[d] += f_costs[d]

    from operator import itemgetter

    # print(f" ### On selecting value for {variable.name} : {d_costs}")
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
        for f in factors:
            if f == factor or f not in costs:
                continue
            # [f for f in factors if f != factor and f in costs]:
            f_costs = costs[f]
            if d not in f_costs:
                continue
            c = f_costs[d]
            sum_cost += c
            msg_costs[d] += c

    # Experimentally, when we do not normalize costs the algorithm takes
    # more cycles to stabilize
    # return {d: c for d, c in msg_costs.items() }

    # Normalize costs with the average cost, to avoid exploding costs
    avg_cost = sum_cost / len(msg_costs)
    normalized_msg_costs = {
        d: c - avg_cost for d, c in msg_costs.items()
    }

    return normalized_msg_costs


def apply_damping(costs_f, prev_costs, damping):
    damped_costs = {}
    if prev_costs is not None:
        for d, c in costs_f.items():
            damped_costs[d] = damping * prev_costs[d] + (1 - damping) * c
        return damped_costs
    return costs_f


def approx_match(costs, prev_costs, stability_coef):
    """
    Check if a cost message match the previous message.

    Costs are considered to match if the variation is bellow STABILITY_COEFF.

    :param costs: costs as a dict val -> cost
    :param prev_costs: previous costs as a dict val -> cost
    :return: True if the cost match
    """
    if prev_costs is None:
        return False

    for d, c in costs.items():
        prev_c = prev_costs[d]
        if prev_c != c:
            delta = abs(prev_c - c)
            if prev_c + c != 0:
                if not ((2 * delta / abs(prev_c + c)) < stability_coef):
                    return False
            else:
                return False
    return True


def _valid_assignments(constraint: Constraint, infinity_value):
    """
    Return a list of all valid assignments for the Constraint
    """
    valid_assignments = []
    for assignment in generate_assignment_as_dict(constraint.dimensions[:]):
        if abs(constraint(**assignment)) != infinity_value:
            valid_assignments.append(assignment)
    return valid_assignments
