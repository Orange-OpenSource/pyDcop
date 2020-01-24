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
DSA : Distributed Stochastic Algorithm
--------------------------------------

Distributed Stochastic Algorithm :cite:`zhang_distributed_2005` is a
synchronous, stochastic, local search DCOP algorithm.

This is the classical synchronous version of DSA ; at each cycle each variable
waits for the value of all its neighbors before computing the potential gain
and making a decision. This means that this implementation is not robust to
message loss.


Algorithm Parameters
^^^^^^^^^^^^^^^^^^^^

**variant**
  'A', 'B' or 'C' ; the variant of the algorithm,
  as defined in :cite:`zhang_distributed_2005` . Defaults to B

**probability**
  probability of changing a value. Defaults to 0.7

**stop_cycle**
  The number of cycle after which the algorithm stops, defaults to `0`
  If not defined (of equals to `0`), the computation never stops.

Example
^^^^^^^

::

    pydcop -t 3 solve -algo dsa  \\
      --algo_param stop_cycle:30 \\
      --algo_param variant:C \\
      --algo_param probability:0.5 \\
     -d adhoc graph_coloring_csp.yaml

    {
      "assignment": {
        "v1": "G",
        "v2": "R",
        "v3": "G"
      },
      "cost": 0,
      "duration": 1.9972785034179688,
      "status": "TIMEOUT"
    }

See Also
^^^^^^^^

:ref:`DSA-tuto<implementation_reference_algorithms_dsatuto>`: for a very simple
implementation of DSA, made for tutorials.

:ref:`A-DSA<implementation_reference_algorithms_adsa>`: for an asynchronous
implementation of DSA.




"""
import logging
import random


from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.infrastructure.computations import Message, VariableComputation, register

from pydcop.computations_graph.constraints_hypergraph import VariableComputationNode
from pydcop.dcop.relations import (
    find_optimum,
    assignment_cost,
    filter_assignment_dict,
    find_optimal,
    optimal_cost_value,
)

HEADER_SIZE = 0
UNIT_SIZE = 1

# Type of computations graph that must be used with dsa
GRAPH_TYPE = "constraints_hypergraph"

# Dsa supports several parameters:
#
#     variant: str
#         The DSA variant to use (A, B or C)
#     probability: float
#         The probability threshold for changing value. Used differently
#         depending on the variant of DSA. See (Zhang, 2005) for details
#     p_mode: str
#         TODO
#     stop_cycle: int
#         The number of cycle after which the computation must stop. If not
#         given, the computation does not stop automatically.


algo_params = [
    AlgoParameterDef("probability", "float", None, 0.7),
    AlgoParameterDef("p_mode", "str", ["fixed", "arity"], "fixed"),
    AlgoParameterDef("variant", "str", ["A", "B", "C"], "B"),
    AlgoParameterDef("stop_cycle", "int", None, 0),
]


def computation_memory(computation: VariableComputationNode) -> float:
    """Return the memory footprint of a DSA computation.

    Notes
    -----
    With DSA, a computation must only remember the current value for each
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
        (n for l in computation.links for n in l.nodes if n not in computation.name)
    )
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
    return UNIT_SIZE + HEADER_SIZE


class DsaMessage(Message):
    def __init__(self, value):
        super().__init__("dsa_value", None)
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return 1

    def __str__(self):
        return "DsaMessage({})".format(self.value)

    def __repr__(self):
        return "DsaMessage({})".format(self.value)

    def __eq__(self, other):
        if type(other) != DsaMessage:
            return False
        if self.value == other.value:
            return True
        return False


class DsaComputation(VariableComputation):
    """
    DSAComputation implements several variants of the DSA algorithm.

    See. the following article for a complete description of DSA:
    'Distributed stochastic search and distributed breakout: properties,
    comparison and applications to constraint optimization problems in sensor
    networks', Zhang Weixiong & al, 2005



    Parameters
    ----------
    comp_def: ComputationDef
        a computation definition. The AlgorithmDef object in this computation
        definition MUST be an algorithm definition with dsa.

    Examples
    --------

    > computation = DsaComputation(
    >     ComputationDef(VariableComputationNode(v1, [c1]),
    >                    AlgorithmDef.build_with_default_param('dsa')))


    See Also
    --------
    algo_params for a list of parameter support by this dsa implementation.

    """

    def __init__(self, comp_def: ComputationDef):
        super().__init__(comp_def.node.variable, comp_def)

        assert comp_def.algo.algo == "dsa"
        assert (comp_def.algo.mode == "min") or (comp_def.algo.mode == "max")

        self.mode = comp_def.algo.mode
        self.probability = comp_def.algo.param_value("probability")
        self.variant = comp_def.algo.param_value("variant")
        self.stop_cycle = comp_def.algo.param_value("stop_cycle")
        self.constraints = comp_def.node.constraints

        if comp_def.algo.param_value("p_mode") == "arity":
            n_count = sum(len(c.dimensions) - 1 for c in self.constraints)
            n = len(comp_def.node.neighbors)
            self.probability = 1 / n_count * 1.2
            self.logger.debug(
                f"Using arity-based threshold : {self.probability} {n_count} {n}"
            )

        # Maps for the values of our neighbors for the current and next cycle:
        self.current_cycle = {}
        self.next_cycle = {}

        if self.variant == "B":
            # In DSA-B, we need to check if there are still some violated
            # constraints, for this we compute the best achievable cost for each
            # constraint:
            self.best_constraints_costs = {
                c.name: find_optimum(c, self.mode) for c in self.constraints
            }

    def on_start(self):
        if not self.neighbors:
            # If a variable has no neighbors, we must select its final value immediately
            # as it will never receive any message.
            value, cost = optimal_cost_value(self._variable, self.mode)
            self.value_selection(value, cost)
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"Select initial value {self.current_value} "
                    f"based on cost function for var {self._variable.name}"
                )
            self.finished()
            self.stop()
        else:
            self.random_value_selection()
            self.logger.debug(
                "DSA starts: randomly select value %s", self.current_value
            )
            self.post_to_all_neighbors(DsaMessage(self.current_value))

            # As everything is asynchronous, we might have received our
            # neighbors values even before starting this algorithm.
            self.evaluate_cycle()

    @register("dsa_value")
    def _on_value_msg(self, variable_name, recv_msg, t):
        if not self._running:
            return
        if variable_name not in self.current_cycle:
            self.current_cycle[variable_name] = recv_msg.value
            self.logger.debug(
                "Receiving value %s from %s", recv_msg.value, variable_name
            )
            self.evaluate_cycle()

        else:
            self.logger.debug(
                "Receiving value %s from %s for the next cycle.",
                recv_msg.value,
                variable_name,
            )
            self.next_cycle[variable_name] = recv_msg.value

    def evaluate_cycle(self):

        if len(self.current_cycle) == len(self.neighbors):

            self.logger.debug(
                "Full neighbors assignment for cycle %s : %s ",
                self.cycle_count,
                self.current_cycle,
            )

            self.current_cycle[self.variable.name] = self.current_value
            assignment = self.current_cycle.copy()
            args_best, best_cost = find_optimal(
                self.variable, assignment, self.constraints, self.mode
            )
            current_cost = assignment_cost(self.current_cycle, self.constraints)
            delta = abs(current_cost - best_cost)
            self.logger.debug(
                f"Current cost {current_cost}, best cost {best_cost} " f"delta {delta}"
            )

            if self.variant == "A":
                self.variant_a(delta, best_cost, args_best)
            elif self.variant == "B":
                self.variant_b(delta, best_cost, args_best)
            elif self.variant == "C":
                self.variant_c(delta, best_cost, args_best)

            self.new_cycle()
            self.current_cycle, self.next_cycle = self.next_cycle, {}

            # Check if this was the last cycle
            if self.stop_cycle and self.cycle_count >= self.stop_cycle:
                self.finished()
                self.stop()
                return

            self.post_to_all_neighbors(DsaMessage(self.current_value))

    def variant_a(self, delta, best_cost, best_values):
        """
        DSA-A value change : only if gain is strictly positive.
        """
        if delta > 0:
            self.logger.debug("Variant A, attempt probabilistic change")
            self.probabilistic_change(best_cost, best_values)
        else:
            self.logger.debug("Variant A, no reason to change")

    def variant_b(self, delta, best_cost, best_values):
        """
        DSA-B value change : only if gain is positive or == 0 but some
        constraints are still violated (i.e. not at their optimal value).
        """
        if delta > 0:
            self.logger.debug("Variant B, attempt probabilistic change")
            self.probabilistic_change(best_cost, best_values)

        elif delta == 0 and self.exists_violated_constraint():
            self.logger.debug("Variant B, attempt probabilistic change")
            if len(best_values) > 1:
                try:
                    best_values.remove(self.current_value)
                except ValueError:
                    pass
            self.probabilistic_change(best_cost, best_values)
        else:
            self.logger.debug("Variant B, no reason to change")

    def variant_c(self, delta, best_cost, best_values):
        """
        DSA-B value change : if gain is <= 0.
        """
        if delta > 0:
            self.logger.debug("Variant C, attempt probabilistic change")
            self.probabilistic_change(best_cost, best_values)

        elif delta == 0:
            if len(best_values) > 1:
                try:
                    best_values.remove(self.current_value)
                except ValueError:
                    pass
            self.probabilistic_change(best_cost, best_values)
        else:
            self.logger.debug("Variant C, no reason to change")

    def probabilistic_change(self, best_cost, best_values):
        """
        Select a new value if we randomly reach the probability threshold.
        """
        if self.probability > random.random():
            self.value_selection(random.choice(best_values), best_cost)
            self.logger.info(
                f"Selecting new value {self.current_value} with cost {self.current_cost}"
            )
        else:
            self.logger.info("No probabilistic value change")

    def exists_violated_constraint(self) -> bool:
        """
        Tells if there is a violated soft constraint regarding the current
        assignment
        :return: a boolean
        """
        for c in self.constraints:
            asgt = self.current_cycle.copy()
            asgt[self.name] = self.current_value
            const = c(**filter_assignment_dict(asgt, c.dimensions))
            if const != self.best_constraints_costs[c.name]:
                return True
        return False
