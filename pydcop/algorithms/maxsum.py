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
from random import choice

from typing import Dict, Union, Tuple, Any, List

from collections import defaultdict

from pydcop.computations_graph.factor_graph import (
    VariableComputationNode,
    FactorComputationNode,
)
from pydcop.dcop.objects import VariableNoisyCostFunc, Variable
from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.dcop.relations import generate_assignment_as_dict
from pydcop.infrastructure.computations import (
    Message,
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
        factor_names = [l.factor_node for l in comp_def.node.links]
        logger.debug(
            "building variable computation {} - {}".format(comp_def.node, factor_names)
        )
        return VariableAlgo(comp_def.node.variable, factor_names, comp_def=comp_def)
    if comp_def.node.type == "FactorComputation":
        logger.debug("building factor computation {}".format(comp_def.node))
        return FactorAlgo(comp_def.node.factor, comp_def=comp_def)


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
    AlgoParameterDef("infinity", "int", None, 10000),
    AlgoParameterDef("stability", "float", None, 0.1),
    AlgoParameterDef("damping", "float", None, 0.0),
    AlgoParameterDef("stability", "float", None, STABILITY_COEFF),
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
        # of list to avoid problem when serializing / deserializing the repr.
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


class FactorAlgo(DcopComputation):
    """
    FactorAlgo encapsulate the algorithm running at factor's node.

    """

    def __init__(
        self,
        factor,
        name=None,
        msg_sender=None,
        infinity=INFINITY,
        stability=STABILITY_COEFF,
        comp_def=None,
    ):
        """
        Factor algorithm (factor can be n-ary).
        Variables does not need to be listed explicitly, they are taken from
        the factor function.

        :param factor: a factor object implementing the factor protocol ,
        :param msg_sender: the object that will be used to send messages to
        neighbors, it must have a post_msg(sender, target_name, name) method.
        """
        name = name if name is not None else factor.name
        super().__init__(name, comp_def)

        assert comp_def.algo.algo == "maxsum"
        assert (comp_def.algo.mode == "min") or (comp_def.algo.mode == "max")

        self._factor = factor
        self.mode = comp_def.algo.mode

        global INFINITY, STABILITY_COEFF
        INFINITY = infinity
        STABILITY_COEFF = stability

        # costs : messages for our variables, used to store the content of the
        # messages received from our variables.
        # v -> d -> costs
        # For each variable, we keep a dict mapping the values for this
        # variable to an associated cost.
        self._costs = {}

        self._msg_sender = msg_sender

        # A dict var_name -> (message, count)
        self._prev_messages = defaultdict(lambda: (None, 0))

        if len(self.variables) <= 1:
            self._is_stable = True
        else:
            self._is_stable = False

        self._valid_assignments_cache = None
        self._valid_assignments()

    @property
    def name(self):
        return self._name

    @property
    def variables(self):
        """
        :return: The list of variables objects the factor depends on.
        """
        return self._factor.dimensions

    @property
    def factor(self):
        return self._factor

    @property
    def is_stable(self):
        return self._is_stable

    def footprint(self):
        return computation_memory(self.computation_def.node)

    def on_start(self):
        msg_count, msg_size = 0, 0

        # Only unary factors (leaf in the graph) needs to send their costs at
        # init.Each leaf factor sends his costs to its only variable.
        # When possible it is better to use a variable with integrated costs
        # instead of a variable with an unary relation representing costs.
        if len(self.variables) == 1:
            self.logger.warning("Sending init costs of unary factor %s", self.name)
            msg_count, msg_size = self._init_msg()

        return {"num_msg_out": msg_count, "size_msg_out": msg_size}

    def _init_msg(self):
        msg_debug = []
        msg_count, msg_size = 0, 0

        for v in self.variables:
            costs_v = self._costs_for_var(v)
            msg_size += self._send_costs(v.name, costs_v)
            msg_count += 1
            msg_debug.append((v.name, costs_v))

        if self.logger.isEnabledFor(logging.DEBUG):
            debug = "Unary factor : init msg {} \n".format(self.name)
            for dest, msg in msg_debug:
                debug += "  * {} -> {} : {}\n".format(self.name, dest, msg)
            self.logger.debug(debug + "\n")
        else:
            self.logger.info(
                "Init messages for %s to %s", self.name, [c for c, _ in msg_debug]
            )

        return msg_count, msg_size

    def _send_costs(self, var_name, costs):
        msg = MaxSumMessage(costs)
        size = msg.size
        self.post_msg(var_name, msg)
        return size

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
        send, no_send = [], []
        debug = ""
        msg_count, msg_size = 0, 0

        # Wait until we received costs from all our variables before sending
        # our own costs
        if len(self._costs) == len(self._factor.dimensions):
            stable = True
            for v in self.variables:
                if v.name != var_name:
                    costs_v = self._costs_for_var(v)
                    same, same_count = self._match_previous(v.name, costs_v)
                    if not same or same_count < SAME_COUNT:
                        debug += "  * SEND {} -> {} : {}\n".format(
                            self.name, v.name, costs_v
                        )
                        msg_size += self._send_costs(v.name, costs_v)
                        send.append(v.name)
                        msg_count += 1
                        self._prev_messages[v.name] = costs_v, same_count + 1
                        self._is_stable = False
                    else:
                        no_send.append(v.name)
                        debug += "  * NO-SEND {} -> " "{} : {}\n".format(
                            self.name, v.name, costs_v
                        )
            self._is_stable = stable
        else:
            debug += (
                "  * Still waiting for costs from all"
                " the variables {}\n".format(self._costs.keys())
            )

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "ON %s -> %s message : %s  \n%s", var_name, self.name, msg.costs, debug
            )
        else:
            self.logger.info(
                "On cost msg from %s, send messages to %s - no " "send %s",
                var_name,
                send,
                no_send,
            )

        return {"num_msg_out": msg_count, "size_msg_out": msg_size}

    def _costs_for_var(self, variable):
        """
        Produce the message for the variable v.

        The content of this message is a table d -> mincost where
        * d is a value of the domain of the variable v
        * mincost is the minimum value of f when the variable v take the 
          value d

        :param variable: the variable we want to send the costs to
        :return: a mapping { value => cost}
        where value is all the values from the domain of 'variable'
        costs is the cost when 'variable'  == 'value'

        """
        costs = {}
        for d in variable.domain:
            # for each value d in the domain of v, calculate min cost (a)
            # where a is any assignment where v = d
            # cost (a) = f(a) + sum( costvar())
            # where costvar is the cost received from our other variables

            mode_opt = INFINITY if self.mode == "min" else -INFINITY
            optimal_value = mode_opt

            for assignment in self._valid_assignments():
                if assignment[variable.name] != d:
                    continue
                f_val = self._factor(**assignment)
                if f_val == INFINITY:
                    continue

                sum_cost = 0
                # sum of the costs from all other variables
                for another_var, var_value in assignment.items():
                    if another_var == variable.name:
                        continue
                    if another_var in self._costs:
                        if var_value not in self._costs[another_var]:
                            # If there is no cost for this value, it means it
                            #  is infinite (as infinite cost are not included
                            # in messages) and we can stop adding costs.
                            sum_cost = mode_opt
                            break
                        sum_cost += self._costs[another_var][var_value]
                    else:
                        # we have not received yet costs from variable v
                        pass

                current_val = f_val + sum_cost
                if (optimal_value > current_val and self.mode == "min") or (
                    optimal_value < current_val and self.mode == "max"
                ):

                    optimal_value = current_val

            if optimal_value != mode_opt:
                costs[d] = optimal_value

        return costs

    def _valid_assignments(self):
        """
        Populates a cache with all valid assignments for the factor
        managed by the algorithm.

        :return: a list of all assignments returning a non-infinite value
        """
        if self._valid_assignments_cache is None:
            self._valid_assignments_cache = []
            all_vars = self._factor.dimensions[:]
            for assignment in generate_assignment_as_dict(all_vars):
                if self._factor(**assignment) != INFINITY:
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
            same = approx_match(costs, prev_costs)
            return same, count
        else:
            return False, 0


class VariableAlgo(VariableComputation):
    def __init__(
        self,
        variable: Variable,
        factor_names: List[str],
        msg_sender=None,
        comp_def: ComputationDef = None,
    ):
        """

        :param variable: variable object
        :param factor_names: a list containing the names of the factors that
        depend on the variable managed by this algorithm
        :param msg_sender: the object that will be used to send messages to
        neighbors, it must have a  post_msg(sender, target_name, name) method.
        """
        super().__init__(variable, comp_def)

        assert comp_def.algo.algo == "maxsum"
        assert (comp_def.algo.mode == "min") or (comp_def.algo.mode == "max")

        self.mode = comp_def.algo.mode

        # self._v = variable.clone()
        # Add noise to the variable, on top of cost if needed
        if variable.has_cost:
            self._v = VariableNoisyCostFunc(
                variable.name,
                variable.domain,
                cost_func=lambda x: variable.cost_for_val(x),
                initial_value=variable.initial_value,
            )
        else:
            self._v = VariableNoisyCostFunc(
                variable.name,
                variable.domain,
                cost_func=lambda x: 0,
                initial_value=variable.initial_value,
            )

        self.var_with_cost = True

        # the currently selected value, will evolve when the algorithm is
        # still running.
        # if self._v.initial_value:
        #     self.value_selection(self._v.initial_value, None)
        #
        # elif self.var_with_cost:
        #     current_cost, current_value =\
        #         min(((self._v.cost_for_val(dv), dv) for dv in self._v.domain ))
        #     self.value_selection(current_value, current_cost)

        # The list of factors (names) this variables is linked with
        self._factors = factor_names

        # The object used to send messages to factor
        self._msg_sender = msg_sender

        # costs : this dict is used to store, for each value of the domain,
        # the associated cost sent by each factor this variable is involved
        # with. { factor : {domain value : cost }}
        self._costs = {}

        self._is_stable = False
        self._prev_messages = defaultdict(lambda: (None, 0))

        self.damping = comp_def.algo.params["damping"]
        self.logger.info("Running maxsum with damping %s", self.damping)

    @property
    def domain(self):
        # Return a copy of the domain to make sure nobody modifies it.
        return self._v.domain[:]

    @property
    def factors(self):
        """
        :return: a list containing the names of the factors which depend on
        the variable managed by this algorithm.
        """
        return self._factors[:]

    def footprint(self):
        return computation_memory(self.computation_def.node)

    def add_factor(self, factor_name):
        """
        Register a factor to this variable.

        All factors depending on a variable MUST be registered so that the
        variable algorithm can send cost messages to them.

        :param factor_name: the name of a factor which depends on this
        variable.
        """
        self._factors.append(factor_name)

    def on_start(self):
        init_stats = self._init_msg()
        return init_stats

    def _init_msg(self):
        # Each variable with integrated costs sends his costs to the factors
        # which depends on it.
        # A variable with no integrated costs simply sends neutral costs
        msg_count, msg_size = 0, 0

        # select our value
        if self.var_with_cost:
            self.value_selection(*self._select_value())
        elif self._v.initial_value:
            self.value_selection(self._v.initial_value, None)
        else:
            self.value_selection(choice(self._v.domain))
        self.logger.info("Initial value selected %s ", self.current_value)

        if self.var_with_cost:
            costs_factors = {}
            for f in self.factors:
                costs_f = self._costs_for_factor(f)
                costs_factors[f] = costs_f

            if self.logger.isEnabledFor(logging.DEBUG):
                debug = "Var : init msgt {} \n".format(self.name)
                for dest, msg in costs_factors.items():
                    debug += "  * {} -> {} : {}\n".format(self.name, dest, msg)
                self.logger.debug(debug + "\n")
            else:
                self.logger.info(
                    "Sending init msg from %s (with cost) to %s",
                    self.name,
                    costs_factors,
                )

            # Sent the messages to the factors
            for f, c in costs_factors.items():
                msg_size += self._send_costs(f, c)
                msg_count += 1
        else:
            c = {d: 0 for d in self._v.domain}
            debug = "Var : init msg {} \n".format(self.name)

            self.logger.info("Sending init msg from %s to %s", self.name, self.factors)

            for f in self.factors:
                msg_size += self._send_costs(f, c)
                msg_count += 1
                debug += "  * {} -> {} : {}\n".format(self.name, f, c)
            self.logger.debug(debug + "\n")

        return {
            "num_msg_out": msg_count,
            "size_msg_out": msg_size,
            "current_value": self.current_value,
        }

    @register("max_sum")
    def _on_maxsum_msg(self, factor_name, msg, t):
        """
        Handling cost message from a neighbor factor.

        :param factor_name: the name of that factor that sent us this message.
        :param msg: a message whose content is a map { d -> cost } where:
         * d is a value from the domain of this variable
         * cost if the minimum cost of the factor when taking value d
        """
        self._costs[factor_name] = msg.costs

        # select our value
        self.value_selection(*self._select_value())

        # Compute and send our own costs to all other factors.
        # If our variable has his own costs, we must sent them back even
        # to the factor which sent us this message, as integrated costs are
        # similar to an unary factor and with an unary factor we would have
        # sent these costs back to the original sender:
        # factor -> variable -> unary_cost_factor -> variable -> factor
        fs = self.factors
        if not self.var_with_cost:
            fs.remove(factor_name)

        msg_count, msg_size = self._compute_and_send_costs(fs)

        # return stats about this cycle:
        return {
            "num_msg_out": msg_count,
            "size_msg_out": msg_size,
            "current_value": self.current_value,
        }

    def _compute_and_send_costs(self, factor_names):
        """
        Computes and send costs messages for all factors in factor_names.

        :param factor_names: a list of names of factors to compute and send
        messages to.
        """
        debug = ""
        stable = True
        send, no_send = [], []
        msg_count, msg_size = 0, 0
        for f_name in factor_names:
            costs_f = self._costs_for_factor(f_name)
            same, same_count = self._match_previous(f_name, costs_f)
            if not same or same_count < SAME_COUNT:
                debug += "  * SEND : {} -> {} : {}\n".format(self.name, f_name, costs_f)
                msg_size += self._send_costs(f_name, costs_f)
                send.append(f_name)
                self._prev_messages[f_name] = costs_f, same_count + 1
                stable = False
                msg_count += 1

            else:
                no_send.append(f_name)
                debug += "  * NO-SEND : {} -> {} : {}\n".format(
                    self.name, f_name, costs_f
                )
        self._is_stable = stable

        # Display sent messages
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Sending messages from %s :\n%s", self.name, debug)
        else:
            self.logger.info(
                "Sending messages from %s to %s, no_send %s", self.name, send, no_send
            )

        return msg_count, msg_size

    def _send_costs(self, factor_name, costs):
        """
        Sends a cost messages and return the size of the message sent.
        :param factor_name:
        :param costs:
        :return:
        """
        msg = MaxSumMessage(costs)
        self.post_msg(factor_name, msg)
        return msg.size

    def _select_value(self) -> Tuple[Any, float]:
        """

        Returns
        -------
        a Tuple containing the selected value and the corresponding cost for
        this computation.
        """

        # If we have received costs from all our factor, we can select a
        # value from our domain.
        if self.var_with_cost:
            # If our variable has it's own cost, take them into account
            d_costs = {d: self._v.cost_for_val(d) for d in self._v.domain}
        else:
            d_costs = {d: 0 for d in self._v.domain}
        for d in self._v.domain:
            for f_costs in self._costs.values():
                if d not in f_costs:
                    # As infinite costs are not included in messages,
                    # if there is not cost for this value it means the costs
                    # is infinite and we can stop adding other costs.
                    d_costs[d] = INFINITY if self.mode == "min" else -INFINITY
                    break
                d_costs[d] += f_costs[d]

        from operator import itemgetter

        if self.mode == "min":
            optimal_d = min(d_costs.items(), key=itemgetter(1))
        else:
            optimal_d = max(d_costs.items(), key=itemgetter(1))

        return optimal_d[0], optimal_d[1]

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

    def _costs_for_factor(self, factor_name):
        """
        Produce the message that must be sent to factor f.

        The content if this message is a d -> cost table, where
        * d is a value from the domain
        * cost is the sum of the costs received from all other factors except f
        for this value d for the domain.

        :param factor_name: the name of a factor for this variable
        :return: the value -> cost table
        """
        # If our variable has integrated costs, add them
        if self.var_with_cost:
            msg_costs = {d: self._v.cost_for_val(d) for d in self._v.domain}
        else:
            msg_costs = {d: 0 for d in self._v.domain}

        sum_cost = 0
        for d in self._v.domain:
            for f in [f for f in self.factors if f != factor_name and f in self._costs]:
                f_costs = self._costs[f]
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

        prev_costs, count = self._prev_messages[factor_name]
        damped_costs = {}
        if prev_costs is not None:
            for d, c in msg_costs.items():
                damped_costs[d] = self.damping * prev_costs[d] + (1 - self.damping) * c
            self.logger.warning("damping : replace %s with %s", msg_costs, damped_costs)
            msg_costs = damped_costs

        return msg_costs
