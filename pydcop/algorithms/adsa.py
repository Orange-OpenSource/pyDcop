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
A-DSA : Asynchronous Distributed Stochastic Algorithm
-----------------------------------------------------

ADSA :cite:`weiss_distributed_2003`
is an asynchronous version of DSA, the Distributed Stochastic Algorithm
:cite:`zhang_distributed_2005` ( stochastic, local search DCOP algorithm.)
Instead of waiting for its neighbors, each variables periodically determines if
it should select a new values, based on the values received from its neighbors.



Algorithm Parameters
^^^^^^^^^^^^^^^^^^^^
**variant**
  'A', 'B' or 'C' ; the variant of the algorithm,
  as defined in :cite:`zhang_distributed_2005` . Defaults to B.

**probability**
  probability of changing a value. Defaults to 0.7.

**period**
  The period between variables activation, in second. Defaults to 0.5.


Example
^^^^^^^

::

    pydcop -t 10 solve --algo adsa \\
      --algo_param variant:B  \\
      --algo_param probability:0.5 \\
      --algo_param period:0.2
      graph_coloring_50.yaml


See Also
^^^^^^^^

:ref:`DSA-tuto<implementation_reference_algorithms_dsatuto>`: for a very simple
implementation of DSA, made for tutorials.

:ref:`A-DSA<implementation_reference_algorithms_dsa>`: for an asynchronous
implementation of DSA.



"""
import logging
import random
from typing import Tuple, Any, List, Dict

from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.dcop.relations import (
    find_optimum,
    assignment_cost,
    filter_assignment_dict,
    optimal_cost_value,
)
from pydcop.infrastructure.computations import (
    VariableComputation,
    register,
    message_type,
    DcopComputation,
)

# Type of computations graph that must be used with dsa
GRAPH_TYPE = "constraints_hypergraph"


def build_computation(comp_def: ComputationDef) -> DcopComputation:
    """Build a DSA computation

    Parameters
    ----------
    comp_def: a ComputationDef object
        the definition of the DSA computation

    Returns
    -------
    MessagePassingComputation
        a message passing computation that implements the DSA algorithm for
        one variable.

    """
    return ADsaComputation(comp_def=comp_def)


algo_params = [
    AlgoParameterDef("period", "float", None, 0.5),
    AlgoParameterDef("probability", "float", None, 0.7),
    AlgoParameterDef("variant", "str", ["A", "B", "C"], "B"),
]


ADsaMessage = message_type("adsa_value", ["value"])


class ADsaComputation(VariableComputation):
    def __init__(self, comp_def):
        super().__init__(comp_def.node.variable, comp_def)

        assert comp_def.algo.algo == "adsa"
        assert (comp_def.algo.mode == "min") or (comp_def.algo.mode == "max")

        self.mode = comp_def.algo.mode
        self.probability = comp_def.algo.param_value("probability")
        self.variant = comp_def.algo.param_value("variant")
        self.period = comp_def.algo.param_value("period")
        self.constraints = comp_def.node.constraints

        self.current_assignment = {}

        if self.variant == "B":
            # In DSA-B, we need to check if there are still some violated
            # constraints, for this we compute the best achievable cost for each
            # constraint:
            self.best_constraints_costs = {
                c.name: find_optimum(c, self.mode) for c in self.constraints
            }

        self._start_handle = None
        self._tick_handle = None

    def on_start(self):
        delay = random.random() * self.period
        self._start_handle = self.add_periodic_action(delay, self.delayed_start)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Add start delayed action ")

    def on_stop(self):
        if self._tick_handle:
            self.remove_periodic_action(self._tick_handle)
        else:
            self.logger.warning(
                f"Stopping a adsa computation {self.variable} that never really started ! "
                "no _tick_handle"
            )

    def on_pause(self, paused: bool):
        if not paused:
            # when resuming (i.e. leaving pause) we can simply drop any pending message
            # as A-DSA is asynchronous and periodic
            self._paused_messages_post.clear()
            self._paused_messages_recv.clear()
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Dropping all message from pause on {self.name}")

    def delayed_start(self):
        self.remove_periodic_action(self._start_handle)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Remove start delayed action %s ", self._start_handle)
        self._start_handle = None

        if not self.neighbors:
            # If a variable has no neighbors, we must select its final value immediately.
            # We also do not need to setup a periodic action.
            if hasattr(self._variable, "cost_for_val"):
                current_cost, value = optimal_cost_value(self._variable, self.mode)
                self.value_selection(value, current_cost)
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"ADSA starts: initial value {self.current_value} "
                        f"based on cost function for var {self._variable.name}"
                    )
            else:
                self.value_selection(random.choice(self.variable.domain), None)
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"ADSA starts: initial random value {self.current_value} "
                        f"for unconstrained variable {self._variable.name}"
                    )
            self.finished()
            self.stop()
        else:
            self._tick_handle = self.add_periodic_action(self.period, self.tick)
            self.random_value_selection()
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    "ADSA starts: randomly select value %s", self.current_value
                )
            self.post_to_all_neighbors(ADsaMessage(self.current_value))

    @register("adsa_value")
    def _on_value_msg(self, variable_name, msg: ADsaMessage, _):
        self.current_assignment[variable_name] = msg.value
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Receiving value %s from %s", msg.value, variable_name)

    def tick(self):
        if self.is_paused:
            return
        # Check if we have a value for all our neighbors
        if len(self.current_assignment) == len(self.neighbors):

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Full neighbors assignment on periodic action %s : %s ",
                    self.cycle_count,
                    self.current_assignment,
                )

            assignment = self.current_assignment.copy()
            args_best, best_cost = self.find_best_values(assignment)

            # if self.current_value is not None:
            assignment[self.variable.name] = self.current_value
            current_cost = assignment_cost(assignment, self.constraints)
            delta = abs(current_cost - best_cost)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Current value {self.current_value}, cost {current_cost}, "
                    f"best cost {best_cost} "
                    f"delta {delta}"
                )
            if self.variant == "A":
                self.variant_a(delta, best_cost, args_best)
            elif self.variant == "B":
                self.variant_b(delta, best_cost, args_best)
            elif self.variant == "C":
                self.variant_c(delta, best_cost, args_best)

        else:
            n = len(self.neighbors)
            c = len(self.current_assignment)

            print(f" {self.name} Still waiting for neighbors values {n-c} out of {n} ")
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Still waiting for neighbors values {n-c} out of {n} "
                )

        # In order to be more resilient to message loss, we send our value even
        # if it did not change.
        self.post_to_all_neighbors(ADsaMessage(self.current_value))

    def variant_a(self, delta, best_cost, best_values):
        """
        DSA-A value change : only if gain is strictly positive.
        """
        if delta > 0:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Variant A, attempt probabilistic change")
            self.probabilistic_change(best_cost, best_values)
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Variant A, no reason to change")

    def variant_b(self, delta, best_cost, best_values):
        """
        DSA-B value change : only if gain is positive or == 0 but some
        constraints are still violated (i.e. not at their optimal value).
        """
        if delta > 0:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Variant B, attempt probabilistic change")
            self.probabilistic_change(best_cost, best_values)

        elif delta == 0 and self.exists_violated_constraint():
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Variant B, attempt probabilistic change")
            if len(best_values) > 1:
                try:
                    best_values.remove(self.current_value)
                except ValueError:
                    pass
            self.probabilistic_change(best_cost, best_values)
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Variant B, no reason to change")

    def variant_c(self, delta, best_cost, best_values):
        """
        DSA-B value change : if gain is <= 0.
        """
        if delta > 0:
            if self.logger.isEnabledFor(logging.DEBUG):
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
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Variant C, no reason to change")

    def probabilistic_change(self, best_cost, best_values):
        """
        Select a new value if we randomly reach the probability threshold.
        """
        if self.probability > random.random():
            self.value_selection(random.choice(best_values), best_cost)
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"Selecting new value {self.current_value} with cost {self.current_cost}"
                )
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.info("No probabilistic value change")

    def find_best_values(self, assignment: Dict[Any, float]) -> Tuple[List[Any], float]:
        """
        Find the best values for our variable, given the current assignment.

        Find the values from the domain of our variable that yield the best
        cost (min or max depending of mode) given the assignment known for our
        neighbors.

        Parameters
        ----------
        assignment:
            The current assignment

        Returns
        -------
        List[Any]
            A list of values from the domain of our variable
        float
            The cost achieved with these values.
        """
        assignment = assignment.copy()

        arg_best, best_cost = None, float("inf")
        if self.mode == "max":
            arg_best, best_cost = None, -float("inf")

        for value in self.variable.domain:
            assignment[self.variable.name] = value
            cost = assignment_cost(assignment, self.constraints)

            # Take into account variable cost, if any
            cost += self.variable.cost_for_val(value)

            if cost == best_cost:
                arg_best.append(value)
            elif (self.mode == "min" and cost < best_cost) or (
                self.mode == "max" and cost > best_cost
            ):
                best_cost, arg_best = cost, [value]

        return arg_best, best_cost

    def exists_violated_constraint(self) -> bool:
        """
        Tells if there is a violated soft constraint regarding the current
        assignment
        :return: a boolean
        """
        assignment = self.current_assignment.copy()
        assignment[self.variable.name] = self.current_value
        for c in self.constraints:
            const = c(**filter_assignment_dict(assignment, c.dimensions))
            if const != self.best_constraints_costs[c.name]:
                return True
        return False
