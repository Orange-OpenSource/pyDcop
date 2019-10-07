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

This is a **asynchronous implementation** of Max-Sum,
where factors and variable send messages
every time they receive a message.
For an synchronous implementation,
see. :ref:`Max-Sum<implementation_reference_algorithms_maxsum>`




Algorithm Parameters
^^^^^^^^^^^^^^^^^^^^


Example
^^^^^^^

::

    pydcop solve -algo amaxsum  \\
     -d adhoc graph_coloring_csp.yaml

FIXME: add results

See Also
^^^^^^^^
:ref:`Max-Sum<implementation_reference_algorithms_maxsum>`: an synchronous implementation of
Max-Sum.


"""


import logging

from collections import defaultdict
from typing import Dict, Any, List

from pydcop.dcop.objects import VariableNoisyCostFunc, Variable
from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.algorithms import maxsum
from pydcop.dcop.relations import generate_assignment_as_dict
from pydcop.infrastructure.computations import (
    DcopComputation,
    VariableComputation,
    register,
)


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

# reuse same algorithms parameters as MaxSum
algo_params = maxsum.algo_params


class MaxSumFactorComputation(DcopComputation):
    """
    FactorAlgo encapsulate the algorithm running at factor's node.

    """

    def __init__(self, comp_def=None):
        assert comp_def.algo.algo == "amaxsum"
        super().__init__(comp_def.node.factor.name, comp_def)
        self.mode = comp_def.algo.mode
        self.factor = comp_def.node.factor
        self.variables = self.factor.dimensions

        # costs : messages for our variables, used to store the content of the
        # messages received from our variables.
        # v -> d -> costs
        # For each variable, we keep a dict mapping the values for this
        # variable to an associated cost.
        self._costs = {}

        self.damping = comp_def.algo.params["damping"]
        self.damping_nodes = comp_def.algo.params["damping_nodes"]
        self.stability_coef = comp_def.algo.params["stability"]
        self.start_messages = comp_def.algo.params["start_messages"]
        self.logger.info(f"Running maxsum with params: {comp_def.algo.params}")

        # A dict var_name -> (message, count)
        self._prev_messages = defaultdict(lambda: (None, 0))

    def footprint(self) -> float:
        return computation_memory(self.computation_def.node)

    def on_start(self):
        # Only unary factors (leaf in the graph) needs to send their costs at
        # init.Each leaf factor sends his costs to its only variable.
        # When possible it is better to use a variable with integrated costs
        # instead of a variable with an unary relation representing costs.
        if len(self.variables) == 1 and self.start_messages in ["leafs", "leafs_vars"]:
            for v in self.variables:
                costs_v = maxsum.factor_costs_for_var(
                    self.factor, v, self._costs, self.mode
                )
                self.post_msg(v.name, maxsum.MaxSumMessage(costs_v))
                self.logger.info(
                    f"Sending init messages from factor {self.name} -> {v.name} : {costs_v}"
                )
        elif self.start_messages == "all":
            for v in self.variables:
                costs_v = maxsum.factor_costs_for_var(
                    self.factor, v, self._costs, self.mode
                )
                self.post_msg(v.name, maxsum.MaxSumMessage(costs_v))
                self.logger.info(
                    f"Sending init messages from factor {self.name} -> {v.name} : {costs_v}"
                )

    def on_pause(self, paused: bool):
        # When resuming a computation, send messages as for start
        if paused:
            return
        # flushing cost table when resuming
        self._costs.clear()
        self._prev_messages.clear()
        if len(self.variables) == 1 and self.start_messages in ["leafs", "leafs_vars"]:
            for v in self.variables:
                costs_v = maxsum.factor_costs_for_var(
                    self.factor, v, self._costs, self.mode
                )
                self.post_msg(v.name, maxsum.MaxSumMessage(costs_v))
                self.logger.info(
                    f"Sending resume messages from factor {self.name} -> {v.name} : {costs_v}"
                )
        elif self.start_messages == "all":
            for v in self.variables:
                costs_v = maxsum.factor_costs_for_var(
                    self.factor, v, self._costs, self.mode
                )
                self.post_msg(v.name, maxsum.MaxSumMessage(costs_v))
                self.logger.info(
                    f"Sending resume messages from factor {self.name} -> {v.name} : {costs_v}"
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
        # our own costs (if works without doing that, but results are worse)
        if len(self._costs) == len(self.factor.dimensions):
            for v in self.variables:
                if v.name != var_name:
                    costs_v = maxsum.factor_costs_for_var(
                        self.factor, v, self._costs, self.mode
                    )

                    prev_costs, count = self._prev_messages[v.name]

                    # Apply damping to computed costs:
                    if self.damping_nodes in ["factors", "both"]:
                        costs_v = maxsum.apply_damping(
                            costs_v, prev_costs, self.damping
                        )

                    # Check if there was enough change to send the message
                    if not maxsum.approx_match(
                        costs_v, prev_costs, self.stability_coef
                    ):
                        # Not same as previous : send
                        self.logger.debug(
                            f"Sending first time from factor {self.name} -> {v.name} : {costs_v}"
                        )
                        self.post_msg(v.name, maxsum.MaxSumMessage(costs_v))
                        self._prev_messages[v.name] = costs_v, 1

                    elif count < maxsum.SAME_COUNT:
                        # Same as previous, but not yet sent SAME_COUNT times: send
                        self.logger.debug(
                            f"Sending {count} time from variable {self.name} -> {v.name} : {costs_v}"
                        )
                        self.post_msg(v.name, maxsum.MaxSumMessage(costs_v))
                        self._prev_messages[v.name] = costs_v, count + 1
                    else:
                        # Same and already sent SAME_COUNT times: no-send
                        self.logger.debug(
                            f"Not sending (similar) from {self.name} -> {v.name} : {costs_v}"
                        )

        else:
            self.logger.debug(
                f" Still waiting for costs from all  the variables {self._costs.keys()}"
            )


class MaxSumVariableComputation(VariableComputation):
    """
    Maxsum Computation for variable.

    Parameters
    ----------
    comp_def: ComputationDef
    """

    def __init__(self, comp_def: ComputationDef = None):
        super().__init__(comp_def.node.variable, comp_def)
        assert comp_def.algo.algo == "amaxsum"

        self.mode = comp_def.algo.mode
        self.damping = comp_def.algo.params["damping"]
        self.damping_nodes = comp_def.algo.params["damping_nodes"]
        self.stability_coef = comp_def.algo.params["stability"]
        self.start_messages = comp_def.algo.params["start_messages"]
        self.logger.info(f"Running amaxsum with params: {comp_def.algo.params}")

        # The list of factors (names) this variables is linked with
        self._factors = [link.factor_node for link in comp_def.node.links]

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

        # costs : this dict is used to store, for each value of the domain,
        # the associated cost sent by each factor this variable is involved
        # with. { factor : {domain value : cost }}
        self._costs = {}  # type: Dict[str, Dict[Any, float]]

        self._prev_messages = defaultdict(lambda: (None, 0))

    def on_start(self) -> None:
        """
        Startup handler for MaxSum variable computations.

        At startup, a variable select an initial value and send its cost to the factors
        it depends on.
        """

        # select our initial value
        if self.variable.initial_value is not None:
            self.value_selection(self.variable.initial_value, None)
        else:
            self.value_selection(
                *maxsum.select_value(self.variable, self._costs, self.mode)
            )
        self.logger.info(f"Initial value selected {self.current_value}")

        if len(self._factors) == 1 and self.start_messages == "leafs":
            # Only send costs if we are a leaf:
            single_factor = self._factors[0]
            costs_f = maxsum.costs_for_factor(
                self.variable, single_factor, self._factors, self._costs
            )
            self.logger.info(
                f"Sending init msg from leaf variable {self.name} to single factor {single_factor} : {costs_f}"
            )
            self.post_msg(single_factor, maxsum.MaxSumMessage(costs_f))

        elif self.start_messages in ["leafs_vars", "all"]:
            # in "leafs_vars" mode, send our costs to all the factors we depends on.
            for f in self._factors:
                costs_f = maxsum.costs_for_factor(
                    self.variable, f, self._factors, self._costs
                )
                self.logger.info(
                    f"Sending init msg from variable {self.name} to factor {f} : {costs_f}"
                )
                self.post_msg(f, maxsum.MaxSumMessage(costs_f))

    def on_pause(self, paused: bool):
        # When resuming a computation, send messages as for start
        if paused:
            return

        # test flush cost table when resuming
        self._costs.clear()
        self._prev_messages.clear()
        if len(self._factors) == 1 and self.start_messages == "leafs":

            # Only send costs if we are a leaf:
            single_factor = self._factors[0]
            costs_f = maxsum.costs_for_factor(
                self.variable, single_factor, self._factors, self._costs
            )
            self.logger.info(
                f"Sending resume msg from leaf variable {self.name} to single factor {single_factor} : {costs_f}"
            )
            self.post_msg(single_factor, maxsum.MaxSumMessage(costs_f))

        elif self.start_messages in ["leafs_vars", "all"]:
            self.logger.warning(f"Resuming computation {self.name}, send costs ")
            # in "leafs_vars" mode, send our costs to all the factors we depends on.
            for f in self._factors:
                costs_f = maxsum.costs_for_factor(
                    self.variable, f, self._factors, self._costs
                )
                self.logger.info(
                    f"Sending resume msg from variable {self.name} to factor {f} : {costs_f}"
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
        self.value_selection(
            *maxsum.select_value(self.variable, self._costs, self.mode)
        )

        # Compute and send our own costs to all other factors.
        # If our variable has his own costs, we must sent them back even
        # to the factor which sent us this message, as integrated costs are
        # similar to an unary factor and with an unary factor we would have
        # sent these costs back to the original sender:
        # factor -> variable -> unary_cost_factor -> variable -> factor
        for f_name in self._factors:
            if f_name == factor_name:
                continue
            costs_f = maxsum.costs_for_factor(
                self.variable, f_name, self._factors, self._costs
            )
            prev_costs, count = self._prev_messages[f_name]

            # Apply damping to computed costs:
            if self.damping_nodes in ["vars", "both"]:
                costs_f = maxsum.apply_damping(costs_f, prev_costs, self.damping)

            # Check if there was enough change to send the message
            if not maxsum.approx_match(costs_f, prev_costs, self.stability_coef):
                # Not same as previous : send
                self.logger.debug(
                    f"Sending first time from variable {self.name} -> {f_name} : {costs_f}"
                )
                self.post_msg(f_name, maxsum.MaxSumMessage(costs_f))
                self._prev_messages[f_name] = costs_f, 1

            elif count < maxsum.SAME_COUNT:
                # Same as previous, but not yet sent SAME_COUNT times: send
                self.logger.debug(
                    f"Sending {count} time from variable {self.name} -> {f_name} : {costs_f}"
                )
                self.post_msg(f_name, maxsum.MaxSumMessage(costs_f))
                self._prev_messages[f_name] = costs_f, count + 1
            else:
                # Same and already sent SAME_COUNT times: no-send
                self.logger.debug(
                    f"Not sending (similar) from {self.name} -> {f_name} : {costs_f}"
                )
