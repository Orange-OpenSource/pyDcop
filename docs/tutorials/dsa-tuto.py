"""

This module contains a very simple implementation of DSA, for demonstration
purpose.

To keep things as simple as possible, we implemented the bare minimum,
and avoided some details you would generally care about:

* no algorithm parameters (threshold, variants, etc.)
* no computation footprint nor message size

"""


from typing import Any, Tuple

from numpy import random

from pydcop.algorithms import ComputationDef
from pydcop.dcop.relations import assignment_cost
from pydcop.infrastructure.computations import VariableComputation, \
    message_type, register

# Type of computations graph that must be used with dsa
GRAPH_TYPE = 'constraints_hypergraph'

DsaMessage = message_type("dsa_value", ["value"])


class DsaTutoComputation(VariableComputation):
    """
    A very simple DSA implementation.

    Parameters
    ----------
    variable: Variable
        an instance of Variable, whose this computation is responsible for
    constraints: an iterable of constraints objects
        The constraints the variables depends on
    computation_definition: ComputationDef
        the definition of the computation, given as a ComputationDef instance.

    """
    def __init__(self, computation_definition: ComputationDef):
        super().__init__(computation_definition.node.variable,
                         computation_definition)

        assert computation_definition.algo.algo == 'dsatuto'

        self.constraints = computation_definition.node.constraints
        self.current_cycle = {}
        self.next_cycle = {}

    def on_start(self):
        self.random_value_selection()
        self.logger.debug(
            "Random value selected at startup : %s ", self.current_value)
        self.post_to_all_neighbors(DsaMessage(self.current_value))

        if self.is_cycle_complete():
            self.evaluate_cycle()

    @register("dsa_value")
    def on_value_msg(self, variable_name, recv_msg, t):
        self.logger.debug('Receiving %s from %s', recv_msg, variable_name)

        if variable_name not in self.current_cycle:
            self.current_cycle[variable_name] = recv_msg.value
            if self.is_cycle_complete():
                self.evaluate_cycle()

        else:  # The message is for the next cycle
            self.next_cycle[variable_name] = recv_msg.value

    def evaluate_cycle(self):

        self.logger.debug('Full neighbors assignment for cycle %s : %s ',
                          self.cycle_count, self.current_cycle)

        arg_min, min_cost = self.compute_best_value()
        self.current_cycle[self.variable.name] = self.current_value
        current_cost = assignment_cost(self.current_cycle, self.constraints)

        self.logger.debug(
            "Evaluate cycle %s: current cost %s - best cost %s",
            self.cycle_count, current_cost, min_cost)

        if current_cost - min_cost > 0 and 0.5 > random.random():
            self.value_selection(arg_min)
            self.logger.debug(
                "Select new value %s for better cost %s ",
                self.cycle_count, min_cost)
        else:
            self.logger.debug(
                "Do not change value %s ", self.current_value)

        self.new_cycle()
        self.current_cycle, self.next_cycle = self.next_cycle, {}
        self.post_to_all_neighbors(DsaMessage(self.current_value))

    def is_cycle_complete(self):
        # The cycle is complete if we received a value from all the neighbors:
        return len(self.current_cycle) == len(self.neighbors)

    def compute_best_value(self) -> Tuple[Any, float]:

        arg_min, min_cost = None, float('inf')
        for value in self.variable.domain:
            self.current_cycle[self.variable.name] = value
            cost = assignment_cost(self.current_cycle, self.constraints)
            if cost < min_cost:
                min_cost, arg_min = cost, value
        return arg_min, min_cost
