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

Distributed Gibbs: A linear-space sampling-based DCOP algorithm
-----------------------------------------------

Distributed Gibbs: A linear-space sampling-based DCOP algorithm
Nguyen, Duc Thien and Yeoh, William and Lau, Hoong Chuin and Zivan, Roie, 2019
Journal of Artificial Intelligence Research, 705--748, 64, 2019

SD-Gibbs works on a Pseudo-tree, which can be built using the
:ref:`distribute<pydcop_commands_distribute>` command
(and is automatically built when using the :ref:`solve<pydcop_commands_solve>` command).

Algorithm Parameters
^^^^^^^^^^^^^^^^^^^^

* **number_of_iterations**: the amount of iterations that is used as termination criterion.
  Defaults to 20

Example
^^^^^^^
::

    pydcop solve --algo sdgibbs --algo_param number_of_iterations:20 tests/instances/graph_coloring1_func.yaml

"""
from random import choice
from typing import Iterable

import copy
import numpy as np

from pydcop.computations_graph.pseudotree import get_dfs_relations
from pydcop.infrastructure.computations import Message, VariableComputation, register
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import (
    NAryMatrixRelation,
    Constraint,
    find_arg_optimal,
    join,
    projection,
)
from pydcop.algorithms import ALGO_STOP, ALGO_CONTINUE, ComputationDef, AlgoParameterDef

GRAPH_TYPE = "pseudotree"

algo_params = [
    AlgoParameterDef("number_of_iterations", "int", None, 20),
]

def build_computation(comp_def: ComputationDef):

    computation = SDgibbsAlgo(comp_def)
    return computation


def computation_memory(*args):
    raise NotImplementedError("SD-Gibbs has no computation memory implementation (yet)")


def communication_load(*args):
    raise NotImplementedError("SD-Gibbs has no communication_load implementation (yet)")


class DGibbsMessage(Message):
    def __init__(self, msg_type, content):
        super(DGibbsMessage, self).__init__(msg_type, content)

    @property
    def size(self):
        # DGibbs messages
        # VALUE: value of the variable of the sender related to 
        #   the current contexts X_i and \bar{X}_i
        # BACKTRACK: delta of local solution quality between the best-response solution 
        #   in the current iteration and 
        #   delta of local solution quality between the solution in the current iteration.

        if self.type == "VALUE":
            return len(self.content)

        elif self.type == "BACKTRACK":
            return len(self.content)

    def __str__(self):
        return f"DGibbsMessage({self._msg_type}, {self._content})"


class SDgibbsAlgo(VariableComputation):
    """
    Sequential Distributed Gibbs algorithm based on the work of:

    Distributed Gibbs: A linear-space sampling-based DCOP algorithm
        Nguyen, Duc Thien and Yeoh, William and Lau, Hoong Chuin and Zivan, Roie, 2019
        Journal of Artificial Intelligence Research, 705--748, 64, 2019
    """

    def __init__(self, comp_def: ComputationDef):

        assert comp_def.algo.algo == "sdgibbs"

        super().__init__(comp_def.node.variable, comp_def)
        self._mode = comp_def.algo.mode  
        assert self._mode in ['min', 'max']  # assert the mode is valid

        # Get the Depth-First-Tree relations based on the DCOP
        self._parent, self._pseudo_parents, self._children, self._pseudo_children = get_dfs_relations(
            self.computation_def.node
        )

        # Filter the relations on all the nodes of the DFS tree to only keep the
        # relation on the on the lowest node in the tree that is involved in the
        # relation.
        self._constraints = self.computation_def.node.constraints
        self.logger.debug(
            f"Constraints for computation {self.name}: {self._constraints} "
        )
        self._neighbors = self.computation_def.node.neighbors

        self._children_separator = {}

        # Reset the list of neighbors of which a message needs to be received
        self._waited_neighbors = copy.deepcopy(self._children)

        # Algorithm specific parameters
        self._number_of_iterations = comp_def.algo.param_value("number_of_iterations")

        # Algorithm specific variables
        self._value_current = None  # d_i
        self._value_previous = None  # \hat{d}_i
        self._value_optimal = None  # d^\ast_i
        self._value_best = None  # \bar{d}_i
        
        self._context_current = None  # X_i
        self._context_best = None  # \bar{X}_i

        self._time_current = 0  # t_i
        self._time_optimal = 0  # t^\ast_i
        self._time_best = 0  # \bar{t}^\ast_i

        self._Delta = 0.0  # \Delta_i
        self._Delta_best = 0.0  # \bar{\Delta}_i

        if self.is_root:
            self._delta = 0.0  # \delta_i
            self._delta_optimal = 0.0  # \delta^\ast_i
            self._delta_best = 0.0  # \bar{\delta}_i

    def footprint(self):
        return computation_memory(self.computation_def.node)

    @property
    def is_root(self):
        return self._parent is None

    @property
    def is_leaf(self):
        return len(self._children) == 0

    def on_start(self):
        """
        Every agent calls procedure Initialize()

        3 di ← ˆdi ← d∗i ← ¯di ← ValInit(xi)
        4 Xi ← ¯Xi ← {(xj, ValInit(xj)) | xj ∈ Ni} 
        5 ti ← t∗i ← ¯t∗ i ← 0
        6 ∆i ← ¯∆i ← 0 
        7 if xi is root then 
        8   δ ← δ∗ ← ¯δ ← 0
        9   Sample()
        10 end
        """
        # Initiate the value for the variable
        # ValInit(xi)
        # Choose the first value within the domain
        initial_value = self._variable.domain.values[0]
        self._value_current = initial_value
        self._value_previous = initial_value
        self._value_optimal = initial_value
        self._value_best = initial_value

        # Initiate the context
        # Give a value for all the variables contained within the constraints
        dimensions = set([d for c in self._constraints for d in c.dimensions])
        dimensions.remove(self._variable)
        current_context = {d.name: d.domain.values[0] for d in dimensions}

        self._context_current = copy.deepcopy(current_context)
        self._context_best = copy.deepcopy(current_context)


        # Check if the problem is a single agent problem
        if self.is_root and self.is_leaf:
            # If single agent problem, terminate the algorithm
            self._initiate_termination()
        elif self.is_root:
            self._sample_and_send()

    def _sample_and_send(self):
        """
        Procedure Sample()
        11 ti ← ti + 1 
        12 dˆi ← di
        13 di ← Sample based on Equation 21
        14 d¯i ← argmaxd? ?i∈Di ?xj , ¯dj?∈ ¯Xi Fij(d? i, ¯dj)
        15 ∆i ← ?xj,dj?∈Xi Fij(di, dj) −Fij( ˆdi, dj)??
        16 ∆¯ i ← ?xj , ¯dj?∈ ¯Xi Fij( ¯di, ¯dj) −Fij( ˆdi, ¯dj)????
        17 Send VALUE (xi, di, ¯di, t∗ i , ¯t∗ i ) to each xj ∈ Ni
        """
        # 11) Increment the counter
        self._time_current += 1

        # CHECK FOR TERMINATION CRITERION
        if self._time_current > self._number_of_iterations:
            self._initiate_termination()

        # 12) Store the previous value
        self._value_previous = self._value_current

        # 13) Generate a sample
        self._value_current = self._generate_sample()

        # 14) Get the value of the best response
        self._value_best = self._get_best_response_value()

        # 15) Calculate Delta value
        self._Delta = self._calculate_Delta()

        # 16) Calculate Delta value for best response
        self._Delta_best = self._calculate_Delta_best()

        # 17) Send VALUE message to all children
        self._send_value_to_neighbors(
            self._variable.name,  # xi
            self._value_current,  # di
            self._value_best,  #\bar{d}i
            self._time_optimal,  # t^\ast_i
            self._time_best,  # \bar{t}^\ast_i
        )

        return None

    def _calculate_Delta(self):
        """
        Calculate the difference in its local solution quality between 
        the solution in the current iteration 
        (i.e., the agent taking on value di and all its neighbors take on 
        values according to its context Xi) and the solution in the 
        previous iteration (i.e., the agent taking on value ˆdi and all its 
        neighbors take on values according to its context Xi).
        """
        current_cost = self._calculate_local_cost(
                self._context_current, 
                {self._variable.name: self._value_current},
        )
        previous_cost = self._calculate_local_cost(
                self._context_current, 
                {self._variable.name: self._value_previous},
        )
        Delta = current_cost - previous_cost
        return Delta

    def _calculate_Delta_best(self):
        """
        Calculate the difference in its local solution quality between 
        the solution in the current iteration 
        (i.e., the agent taking on value di and all its neighbors take on 
        values according to its context Xi) and the solution in the 
        previous iteration (i.e., the agent taking on value ˆdi and all its 
        neighbors take on values according to its context Xi).
        """
        best_cost = self._calculate_local_cost(
                self._context_best, 
                {self._variable.name: self._value_best},
        )
        previous_best_cost = self._calculate_local_cost(
                self._context_best, 
                {self._variable.name: self._value_previous},
        )
        Delta_best = best_cost - previous_best_cost
        return Delta_best

    def _get_best_response_value(self):
        """
        Return the best value for the best response context
        
        14) \bar{d}_i ← argmax_{d'_i ∈ D_i} Sum_{<x_j, \bar{d}_j> ∈ \bar{X}_i}  F_{ij}(d'_i, \bar{d}_j)
        """
        # utilities = []
        # for v in self._variable.domain.values:
        #     local_best_cost = self._calculate_local_cost(
        #         self._context_best, 
        #         {self._variable.name: v},
        #     )
        #     utilities.append([local_best_cost, v])
        # ONELINER
        utilities = [[self._calculate_local_cost(self._context_best, {self._variable.name: v},), v]
            for v in self._variable.domain.values]

        max_index = np.argmax([u[0] for u in utilities])
        value_best_response = utilities[max_index][1]

        return value_best_response

    
    def _generate_sample(self):
        """
        Generate a sample based on the current context
        Sample based on Equation 21

        P(xi | xj in X {xi}) = 1/Z e^(Sum_{xj, dj in Xi} F_{ij}(di, dj))
        """
        # Sample based on the probability defined in equation 21
        # Note that the normalization constant Z doesn't need to be known
        probabilities = []
        for v in self._variable.domain.values:
            local_cost = self._calculate_local_cost(
                self._context_current, 
                {self._variable.name: v},
            )
            probabilities.append(np.exp(local_cost))
        
        normalization_constant = np.sum(probabilities)
        normalized_probabities = np.array(probabilities) / normalization_constant

        # Generate the sample
        sample_value = np.random.choice(
            self._variable.domain.values,
            p=normalized_probabities,
        )
        
        return sample_value

    def select_value_and_finish(self, value, cost) -> None:
        """
        Select a value for this variable.

        Parameters
        ----------
        value: any (depends on the domain)
            the selected value
        cost: float
            the local cost for this value

        """
        self.value_selection(value, cost)
        self.stop()
        self.finished()
        self.logger.info(f"Value selected at {self.name} : {value} - {cost}")

    # HELPER METHODS
    def _send_message_to_all_neighbors(self, content, message_type) -> None:
        """
        Send a message of 'message_type' to all neighbors of the agent
        """
        # Create the message
        msg = DGibbsMessage(message_type, content)
        # Send to all neighbors
        for neighbor in self._neighbors:
            self.post_msg(target=neighbor, msg=msg)

    def _send_message_to_parent(self, content, message_type) -> None:
        """
        Send a message of 'message_type' to the parent of the agent
        """
        # Create the message
        msg = DGibbsMessage(message_type, content)
        # Send to parent
        self.post_msg(target=self._parent, msg=msg)

    # VALUE MESSAGE RELATED
    @register('VALUE')
    def _on_value_message(self, variable_name, recv_msg, t) -> None:
        """
        Procedure When Received VALUE(xs, ds, ¯ds, t∗s, ¯t∗ s)
        18 Update ?xs, d? s? ∈ Xi with (xs, ds)
        19 if xs ∈ PPi ∪ {Pi} then 
        20   Update ?xs, d? s? ∈ ¯Xi with (xs, ¯ds) 
        21 else 
        22   Update ?xs, d? s? ∈ ¯Xi with (xs, ds)
        23 end 
        24 if xs = Pi then 
        25   if ¯t∗s ≥ t∗s and ¯t∗ s > max{t∗ i , ¯t∗ i } then
        26     d∗i ← ¯di
        27     t¯∗i ← ¯st∗
        28   else if t∗s ≥ ¯t∗ s and t∗ s > max{t∗ i , ¯t∗ i } then
        29   d∗i ← di
        30   t∗i ← t∗s
        31   end
        32   Sample()
        33   if xi is a leaf then
        34     Send BACKTRACK (xi,∆i, ¯∆i) to Pi
        35   end 
        36 end

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: DGibbsMessage
            received message
        t: int
            message timestamp
        """
        self.logger.debug(
            f"{self.name}: on value message from {variable_name} : '{recv_msg}' at {t}"
        )

        # Get the contents of the message
        [received_variable_name, 
         received_value_current, 
         received_value_best, 
         received_time_optimal, 
         received_time_best] = recv_msg.content

        # 18) Update the current context
        self._context_current.update({received_variable_name: received_value_current})

        # 19) Check if message from parent or pseudo-parent
        if variable_name in self._pseudo_parents + [self._parent]:
            # 20) Update the best context based on the best response
            self._context_best.update({received_variable_name: received_value_best})
        else:
            # 22) Update the best context based on the current value
            self._context_best.update({received_variable_name: received_value_current})


        # 24) Check if the message is from the parent
        if variable_name == self._parent:
            # 25) Check the received time indexes
            check_received_times = received_time_best >= received_time_optimal
            check_compare_local_times = received_time_best > np.max([
                self._time_optimal,
                self._time_best,
            ])

            check_received_times2 = received_time_optimal >= received_time_best
            check_compare_local_times2 = received_time_optimal > np.max([
                self._time_optimal,
                self._time_best,
            ])
            if check_received_times and check_compare_local_times:
                # 26) Update the best value for complete assignment/solution
                self._value_optimal = self._value_best
                # 27) Update the time index for best value
                self._time_best = received_time_best
            elif check_received_times2 and check_compare_local_times2:
                # 29) Update the best value
                self._value_optimal = self._value_current
                # 30) Update the time index
                self._time_optimal = received_time_optimal

            # 32) Sample based on the current update
            self._sample_and_send()

            # 33) Check if the agent is a leaf
            if self.is_leaf:
                self._send_backtrack_to_parent(
                    self._variable.name,
                    self._Delta,
                    self._Delta_best,
                )

    def _send_backtrack_to_parent(self, variable_name, Delta, Delta_best) -> None:
        """
        Send a BACKTRACK message to the parent
        """
        content = (variable_name, Delta, Delta_best)
        self._send_message_to_parent(content=content, message_type='BACKTRACK')


    def _send_value_to_neighbors(self, variable_name, value_current, value_best_response, time_best, time_best_response) -> None:
        """
        Send the VALUE message to all the neighbors of the agent
        """
        # Create and send the message
        content = (
            variable_name,
            value_current,
            value_best_response,
            time_best,
            time_best_response
        )
        self._send_message_to_all_neighbors(
            content=content, 
            message_type='VALUE',
        )

    def _calculate_local_cost(self, context, sample):
        """
        Calculate the local cost of the context and sample based on
        Line 14 from Algorithm 1: l^k(a^t_k, d)
        """
        if context is None:
            complete_context = sample
        else:
            complete_context =  dict(context, **sample)

        total_cost = 0.0
        for c in self._constraints:
            total_cost += c.get_value_for_assignment(
                {variable_name: complete_context[variable_name] 
                for variable_name in c.scope_names}
            )

        # Algorithm optimizes the total cost
        # If the mode is minimization, return the negative cost
        if self._mode == 'min':
            total_cost *= -1

        return total_cost

    @register('BACKTRACK')
    def _on_backtrack_message(self, variable_name, recv_msg, t) -> None:
        """
        Message handler for BACKTRACK messages.

        Procedure When Received BACKTRACK(xs,∆s, ¯∆s)
        37 ∆i ← ∆i +∆s 
        38 ∆¯ i ← ¯∆i + ¯∆s
        39 if Received BACKTRACK messages from all children in this iteration then 
        40   Send BACKTRACK (xi,∆i, ¯∆i) to Pi
        41   if xi is root then
        42     δ¯← δ + ¯∆i
        43     δ ← δ +∆i
        44     if δ ≥ ¯δ and δ > δ∗ then
        45      δ∗ ← δ
        46      d∗i ← di
        47      t∗i ← ti
        48     else if ¯δ ≥ δ and ¯δ > δ∗ then
        49       δ∗ ← ¯δ
        50       d∗i ← ¯di
        51       t¯∗i ← ti
        52     end
        53     Sample()
        54   end 
        55 end

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: DGibbsMessage
            received message
        t: int
            message timestamp
        """
        self.logger.debug(
            f"{self.name}: on BACKTRACK message from {variable_name} : '{recv_msg}' at {t}"
        )

        try:
            self._waited_neighbors.remove(variable_name)
        except ValueError as e:
            self.logger.error(
                f"Unexpected BACKTRACK message from {variable_name} on {self.name} : {recv_msg} "
            )
            raise e

        # Get the contents
        [received_variable_name, 
         received_Delta, 
         received_Delta_best] = recv_msg.content

        # 37) Update the Delta 
        self._Delta += received_Delta
        # 38) Update the Delta best
        self._Delta_best += received_Delta_best

        # 39) Check if a BACKTRACK message has been received from all children
        #  in this iteration
        # Check if all messages have been received
        check_all_backtrack_received = len(self._waited_neighbors) == 0
        if check_all_backtrack_received:
            # Reset the list of waited neighbors
            self._waited_neighbors = copy.deepcopy(self._children)

            # 40) Send a backtrack message to the parent
            self._send_backtrack_to_parent(
                self._variable,
                self._Delta,
                self._Delta_best
            )
            # 41) Check if the current agent is root
            if self.is_root:
                # 42) Update the best response shifted utility
                self._delta_best = self._delta + self._Delta_best
                # 43) Update the shifted utility
                self._delta += self._Delta
                # 44) Check the shifted utilities
                if self._delta >= self._delta_best and self._delta > self._delta_optimal:
                    # 45) Update the optimal shifted utility
                    self._delta_optimal = self._delta
                    # 46) Update the optimal value
                    self._value_optimal = self._value_current
                    # 47) Update the time index for the optimal value
                    self._time_optimal = self._time_current
                elif self._delta_best >= self._delta and self._delta_best > self._delta_optimal:
                    # 49) Update the optimal shifted utility
                    self._delta_optimal = self._delta_best
                    # 50) Update the optimal value
                    self._value_optimal = self._value_best
                    # 51) Update the time index for the best value
                    self._time_best = self._time_current

                # 53) Generate and send a new sample
                self._sample_and_send()

    def _initiate_termination(self) -> None:
        """
        Initiate the termination of the algorithm
        """
        optimal_cost = self._calculate_local_cost(
            context = self._context_best,
            sample = {
                self._variable.name: self._value_optimal
            }
        )
        self.select_value_and_finish(self._value_optimal, optimal_cost)
    

    def _print_variables(self):
        """
        Print all variables for debugging purposes
        """
        print(f'-------- {self.name} --------')
        print(f'Current context: {self._context_current}')
        print(f'Best context:    {self._context_best}')
        print(f'')
        print(f'Time current: {self._time_current}')
        print(f'')
        print(f'Value previous:  {self._value_previous}')
        print(f'Value current:   {self._value_current}')
        print(f'Value best:      {self._value_best}')
        print(f'Value optimal:   {self._value_optimal}')
        print(f'')
        print(f'Delta:        {self._Delta}')
        print(f'Delta best:   {self._Delta_best}')
        print(f'')
        if self.is_root:
            print(f'delta:        {self._delta}')
            print(f'delta best:   {self._delta_best}')
            print(f'')
        print(f'Time optimal: {self._time_optimal}')
        print(f'Time best:    {self._time_best}')
        print(f'-----------------------------')
        print(f'')
