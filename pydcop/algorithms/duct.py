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

DUCT: Distributed Upper Confidence Tree
-----------------------------------------------
Brammert Ottens, Christos Dimitrakakis, Boi Faltings. 
DUCT: An Upper Confidence Bound Approach to Distributed Constraint Optimisation Problems
ACM Transactions on Intelligent Systems and Technology, ACM, In press, 8 (5), 
pp.1 - 27. 10.1145/3066156 . hal-01593215

Algorithm Parameters
^^^^^^^^^^^^^^^^^^^^

* **epsilon**: controls how close we wish to be to the optimal solution
  Defaults to 0.6
* **delta**: roughly seen as the probability that the algorithm will give up without having found an epsilon-optimal solution
  Defaults to 0.1
* **bound_method**: bound method selection used during sample selection. Possible values 'naive', 'recursive'.
  Defaults to 'recursive'
* **lambda**: the length of the path to the deepest leaf node
  Defaults to 1

Example
^^^^^^^
::

    pydcop solve --algo duct tests/instances/graph_coloring1_func.yaml

"""
# Generic imports
import copy
import numpy as np
from random import choice
from itertools import product

# pyDCOP imports
from pydcop.computations_graph.pseudotree import get_dfs_relations
from pydcop.infrastructure.computations import Message, VariableComputation, register
from pydcop.algorithms import ALGO_STOP, ALGO_CONTINUE, ComputationDef, AlgoParameterDef

# Define graph type
GRAPH_TYPE = "pseudotree"

# Define the algorithm parameters
algo_params = [
    # controls how close we wish to be to the optimal solution
    AlgoParameterDef("epsilon", "float", None, 0.6),
    # roughly seen as the probability that the algorithm will 
    # give up without having found an epsilon-optimal solution  
    AlgoParameterDef("delta", "float", None, 0.1),  
    # bound method selection used during sample selection
    AlgoParameterDef("bound_method", "str", ['naive', 'recursive'], 'recursive'),
    # the length of the path to the deepest leaf node
    AlgoParameterDef("lambda", "int", None, 1),  
]

def build_computation(comp_def: ComputationDef):
    computation = DuctAlgo(comp_def)
    return computation

def computation_memory(*args):
    raise NotImplementedError("DUCT has no computation memory implementation (yet)")

def communication_load(*args):
    raise NotImplementedError("DUCT has no communication_load implementation (yet)")


class DuctMessage(Message):
    def __init__(self, msg_type, content):
        super(DuctMessage, self).__init__(msg_type, content)

    @property
    def size(self):
        if self.type == "COST":
            # Single value for every value message from parent
            # combined with the bound of the agent
            return len(self.content)

        elif self.type == "CONTEXT":
            # CONTEXT message is a value assignment 
            # for each var in the separator of the sender
            return len(self.content)

        elif self.type == "FINISHED":
            # Contains the final context 
            return len(self.content)

        elif self.type == "LOCAL_BOUND":
            # Contains the upper bound of the local constraints
            # adjusted for the lower bound
            return 1

        elif self.type == "GLOBAL_BOUND":
            # Contains the upper bound of the global constraints 
            # (from the root) adjusted for the lower bound
            return 1

    def __str__(self):
        return f"DuctMessage({self._msg_type}, {self._content})"


class Samples():
    """
    Handling of all properties and methods required 
    for the definition of the counters and 
    the optimal costs (by context)
    """

    def __init__(self, values: list, lambda_context: int):
        # List that holds all samples
        self._samples = []
        # All possible values for the local variable
        self._values = values
        # The length of the path to the deepest leaf node
        self._lambda = lambda_context

    def add_sample(self, context: dict, value, utility: float) -> None:
        """Add a sample to the list of samples

        :param context: complete assignemt for all variables within the seperator of the agent
        :param value: current value of the local variable
        :param utility: utility value for the combined context and (local) value
        """
        self._samples.append([copy.deepcopy(context), value, utility])

    # PROPERTIES OF THE COLLECTED SAMPLES
    def get_optimal_cost_for_context(self, context: dict) -> [float, float]:
        """Returns the optimal cost for the context"""
        sampled_values = self.get_all_sampled_values(context)

        optimal_costs = []
        for _value in sampled_values:
            optimal_cost = self.get_optimal_cost_for_value_under_context(
                context, _value
            )
            optimal_costs.append([optimal_cost, _value])

        opt_index = np.argmin([c[0] for c in optimal_costs])
        optimal_cost, optimal_value = optimal_costs[opt_index]

        return optimal_cost, optimal_value

    def get_optimal_cost_for_value_under_context(self, context: dict, value) -> float:
        """Returns the optimal cost for the value under context (\hat{\mu}^{t}_{a,d})"""
        costs = []
        for _context, _value, cost in self._samples:
            if (context is None or np.all([_context.get(k, None) == v for k,v in context.items()])) and _value == value:
                costs.append(cost)
        if costs:
            optimal_cost = np.min(costs)
        else:
            optimal_cost = np.inf
        return optimal_cost

    def get_count_for_value_under_context(self, context: dict, value) -> int:
        """Returns the count for the value under context"""
        count = 0
        for _context, _value, _ in self._samples:
            if (context is None or np.all([_context.get(k, None) == v for k,v in context.items()])) and _value == value:
                count += 1
        return count

    def get_count_for_context(self, context: dict) -> int:
        """Returns the count for the context"""
        count = 0
        for _context, _, _ in self._samples:
            if context is None or np.all([_context.get(k, None) == v for k,v in context.items()]):
                count += 1
        return count
    
    def get_all_sampled_values(self, context: dict) -> set:
        """Returns all sampled values for the context"""
        values = []
        for _context, _value, _ in self._samples:
            if context is None or np.all([_context.get(k, None) == v for k,v in context.items()]):
                values.append(_value)
        sampled_values = set(values)
        return sampled_values

    # CALCULATION OF BOUNDS
    def get_local_confidence_bound_for_value_under_context(self, context: dict, value) -> float:
        """Returns the local confidence bound for value under context (based on Equation 7)"""
        count_context = self.get_count_for_context(context)
        count_value = self.get_count_for_value_under_context(context, value)

        if count_context == 0:
            lcb = np.inf
        else:
            lcb = np.sqrt(2 * self._lambda * np.log(count_context) / count_value)
        return lcb


class DuctAlgo(VariableComputation):
    """
    Implementation of the Distributed Upper Confidence Tree (DUCT) algorithm.

    Based on the work of
    Brammert Ottens, Christos Dimitrakakis, Boi Faltings. 
    DUCT: An Upper Confidence Bound Approach to Distributed Constraint Optimisation Problems
    ACM Transactions on Intelligent Systems and Technology, ACM, In press, 8 (5), 
    pp.1 - 27. 10.1145/3066156 . hal-01593215
    """

    def __init__(self, comp_def: ComputationDef):
        # Check if the correct algorithm is executed
        assert comp_def.algo.algo == "duct"

        super().__init__(comp_def.node.variable, comp_def)
        self._mode = comp_def.algo.mode
        assert self._mode in ['min', 'max']  # assert the mode is valid

        self._parent, self._pseudo_parents, self._children, self._pseudo_children = get_dfs_relations(
            self.computation_def.node
        )

        # Filter the relations on all the nodes of the DFS tree to only keep the
        # relation on the on the lowest node in the tree that is involved in the relation.
        self._constraints = []
        descendants = self._pseudo_children + self._children
        self.logger.debug(f"Descendants for computation {self.name}: {descendants} ")

        constraints = list(comp_def.node.constraints)
        for r in comp_def.node.constraints:
            # filter out all relations that depends on one of our descendants
            names = [v.name for v in r.dimensions]
            for descendant in descendants:
                if descendant in names:
                    constraints.remove(r)
                    break
        self._constraints = constraints
        self.logger.debug(
            f"Constraints for computation {self.name}: {self._constraints} "
        )

        self._children_separator = {}

        self._waited_children = []
        if not self.is_leaf:
            # If we are not a leaf, we must wait for the LOCAL_BOUND messages 
            # from our children.
            # This must be done in __init__ and not in on_start because we
            # may get an util message from one of our children before
            # running on_start, if this child computation start faster of
            # before us
            self._waited_children = list(self._children)

        # Algorithm specific variables
        self._current_value = None  # d_i
        self._current_context = None  # X_i
        self._current_local_cost = None  # l(a, d)
        self._current_cost = None  # y_i = l(a, d) + sum(y_c) for all children

        self._child_costs = []
        self._child_bounds = []
        self._parent_finished = False
        self._optimal_bound = np.inf

        # normalization related
        # cost_normalized = (cost - local_lower_bound) / global_upper_bound
        self._normalization_finished = False
        self._local_lower_bound = None
        self._global_upper_bound = 0.0

        # Algorithm specific parameters
        # Table 1 from paper
        # algorithm| lambda_a    | sampling
        # ---------|-------------|----------
        # DUCT-A   | 1           | naive
        # DUCT-B   | path length | naive
        # DUCT-C   | 1           | recursive
        # DUCT-D   | path length | recursive
        # controls how close we wish to be to the optimal solution
        self._epsilon = comp_def.algo.param_value("epsilon")  
        # roughly seen as the probability that the algorithm 
        # will give up without having found an epsilon-optimal solution
        self._delta = comp_def.algo.param_value("delta")  
        self._bound_method = comp_def.algo.param_value("bound_method")
        # the length of the path to the deepest leaf node
        self._lambda = comp_def.algo.param_value("lambda")  

        # The Samples class handles all properties
        # and methods required for the definition of
        # the counters and the optimal costs (by context)
        self._samples = Samples(
            values=self._variable.domain.values,
            lambda_context=self._lambda, 
        )

    def footprint(self):
        return computation_memory(self.computation_def.node)

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    def on_start(self) -> None:
        """Execute initial processes, called during start of algorithm
        
        Start calculation of the upper bounds of the constraints,
        as the solver assumes the global utility to be bounded to [0,1]
        
        The normalization process starts from the leaves.
        """
        if self.is_leaf:
            # Calculate the local constraints bounds for normalization
            self._local_lower_bound, local_upper_bound = self._calculate_constraint_bounds()
            # Send the message to the parent
            self._send_message_to_parent(content=local_upper_bound, message_type='LOCAL_BOUND')

    def _initiate_optimization(self) -> None:
        """Start the optimization if the agent is the root
        
        The root starts with sending the initial CONTEXT message
        All other agents wait to receive a CONTEXT message
        """
        if self.is_root:
            self._parent_finished = True
            self._sample_and_send(None)
        else:
            self._parent_finished = False

    def _calculate_constraint_bounds(self) -> [float, float]:
        """Return the lower and upper bound of the (local) constraints
        
        Procedure according to Section 3.3 (exhaustive local search).
        NOTE this assumes that the domains of the neighbors are known.
        """
        # Create a list of all possible contexts (of the (pseudo-)parents)
        relevant_dimensions = [
            d for c in self._constraints
            for d in c.dimensions
            if d.name != self._variable.name
        ]
        variable_names = [d.name for d in relevant_dimensions]
        variable_domains = [d.domain for d in relevant_dimensions]

        if not variable_names:
            # Only dependent on local variables
            context_list = [None]
        else:
            # Fill a list with all permutations of the variable domains
            domain_permutations = [v for v in product(*variable_domains)]
            context_list = [
                {variable_names[i]:v for i, v in enumerate(values)}
                for values in domain_permutations
            ]

        # Perform an exhaustive search over all possible contexts
        cost_buffer = []
        for context in context_list:
            for value in self._variable.domain:
                local_cost = self._calculate_local_cost(context, value)
                cost_buffer.append(local_cost)

        # Get the bounds
        lower_bound = np.array(cost_buffer).min()
        upper_bound = np.array(cost_buffer).max()

        return lower_bound, upper_bound

    def _sample_and_send(self, context: dict) -> None:
        """Sample based on the context and send a CONTEXT message to all children"""
        # Generate a sample
        sample, bound_value = self._generate_sample(context)

        # Store the value and bound
        self._current_value = sample[self._variable.name]

        self._optimal_bound = bound_value

        # Append the sample to the context
        if context is None:
            complete_context = sample
        else:
            complete_context = dict(context, **sample)

        # Send the context to all children
        # and wait for them to send a COST message back
        self._get_total_child_costs(complete_context)

    def _calculate_bound(self, context: dict, value) -> float:
        """Returns the value of the bound based on the bound method"""
        if self._bound_method == 'naive':
            bound = self._calculate_naive_bound(context, value)
        elif self._bound_method == 'recursive':
            bound = self._calculate_recursive_bound(context, value)
        return bound

    def _calculate_recursive_bound(self, context: dict, value) -> float:
        """Returns the bound by the recursive method (Equation 9)"""
        naive_bound = self._calculate_naive_bound(context, value)
        local_cost = self._calculate_local_cost(context, value)
        child_bounds = np.sum(self._child_bounds)

        # NOTE: Equation 9 states 'max' instead of 'min'
        # ASSUMPTION: typo in the paper
        bound = np.min([naive_bound, local_cost + child_bounds])
        return bound

    def _calculate_naive_bound(self, context: dict, value) -> float:
        """Returns the bound by the naive method (Equation 8)"""
        if self.is_leaf:
            bound = self._calculate_local_cost(context, value)
        else:
            lcb = self._samples.get_local_confidence_bound_for_value_under_context(
                context, value)
            opt_cost = self._samples.get_optimal_cost_for_value_under_context(
                context, value)
            if opt_cost == np.inf:
                bound = np.inf
            else:
                bound = opt_cost - lcb
        return bound

    def _generate_duct_sample(self, context: dict) -> [float, float]:
        """Returns the sample value and bound based on Algorithm 3"""
        # Get all the variables that have not been sampled yet
        previously_sampled_values = self._samples.get_all_sampled_values(context)
        unsampled_values = set(self._variable.domain.values) - previously_sampled_values
        all_values_sampled = len(unsampled_values) == 0
        if all_values_sampled:  # line 1
            # line 2
            # using S_a^t Equation 10
            # Loop over all values in S_a^t and calculate the bound B_{a, d}^t
            bounds = []
            for value in self._variable.domain.values:
                # Calculate the bound
                bound = self._calculate_bound(context, value)

                # Check if the value is in S_a^t
                opt_cost = self._samples.get_optimal_cost_for_value_under_context(
                    context, value,
                )
                local_cost = self._calculate_local_cost(context, value)
                is_possible_sample = bound != local_cost + opt_cost
                
                # Add the value to the list
                if is_possible_sample:
                    bounds.append([bound, value])
                else:
                    bounds.append([np.inf, value])

            # Get the optimal value
            opt_index = np.argmin([b[0] for b in bounds])
            bound_value, sample_value = bounds[opt_index]
        else:
            # line 4 (and onwards)
            # random sampling of the untried values
            sample_value = self._generate_random_sample(
                context, list(unsampled_values),
            )
            bound_value = np.inf  # indicates first sample for value
        
        return sample_value, bound_value

    def _generate_random_sample(self, context: dict, possible_values=None) -> float:
        """Returns a random sample from the possible values, if the value is feasible"""
        # Handle the arguments
        if possible_values is None:
            # all values within the domain
            possible_values = self._variable.domain.values

        # Generate a sample
        sample_value = np.random.choice(possible_values)
        # Check if the value is feasible, else pick another value
        sample_utility = self._calculate_local_cost(context, sample_value)

        while (sample_utility == np.inf) and (len(possible_values) > 0):
            # Remove the sample value from the list of possible values
            possible_values.remove(sample_value)
            # Pick a new value
            sample_value = np.random.choice(possible_values)
            # Calculate the new local cost
            sample_utility = self._calculate_local_cost(context, sample_value)

        return sample_value

    def _generate_sample(self, context: dict) -> [dict, float]:
        """Generate a sample based on the context, returns both the sample as the bound"""
        # Generate the sample
        sample_value, bound_value = self._generate_duct_sample(context)

        # Construct the sample (context)
        sample = {self._variable.name: sample_value}
        return sample, bound_value

    def select_value_and_finish(self, value, cost: float) -> None:
        """Select a value for the local variable.

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
    def _send_message_to_all_children(self, content, message_type: str) -> None:
        """Send a message of message_type to all children"""
        # Create the message
        msg = DuctMessage(message_type, content)
        # Send to all children
        for child in self._children:
            self.post_msg(target=child, msg=msg)

    def _send_message_to_parent(self, content, message_type: str) -> None:
        """Send a message of message_type to the parent"""
        # Create the message
        msg = DuctMessage(message_type, content)
        # Send to parent
        self.post_msg(target=self._parent, msg=msg)

    # BOUND MESSAGES RELATED
    @register('LOCAL_BOUND')
    def _on_local_bound_message(self, variable_name: str, recv_msg: DuctMessage, t: int) -> None:
        """Message handler for LOCAL_BOUND messages

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: DuctMessage
            received message
        t: int
            message timestamp
        """
        self.logger.debug(f"LOCAL_BOUND from {variable_name} : {recv_msg.content} at {t}")

        # accumulate cost messages until we got the total cost from all our children
        try:
            self._waited_children.remove(variable_name)
        except ValueError as e:
            self.logger.error(
                f"Unexpected LOCAL_BOUND message from {variable_name} on {self.name} : {recv_msg} "
            )
            raise e

        # process the upper bound
        self._global_upper_bound += recv_msg.content

        # Check if all messages have been received
        if len(self._waited_children) == 0:
            # Calculate the local constraints bounds for normalization
            self._local_lower_bound, local_upper_bound = self._calculate_constraint_bounds()
        
            # Combine the received bounds with the local bounds
            # adjusted by the lowerbound
            self._global_upper_bound += local_upper_bound - self._local_lower_bound 

            if self.is_root:
                self.logger.debug(f"Normalization process finished for root")
                # Flag the normalization as completed
                self._normalization_finished = True
                # The root can combine all bounds and propagate the global bounds 
                # down the tree by sending it to its children
                self._send_message_to_all_children(
                    content=self._global_upper_bound, 
                    message_type='GLOBAL_BOUND',
                )
                # Continue by initiating the optimization
                self._initiate_optimization()
            else:
                # Other agents send the upper bound to its parent
                self._send_message_to_parent(
                    content=self._global_upper_bound, 
                    message_type='LOCAL_BOUND',
                )
            
    @register('GLOBAL_BOUND')
    def _on_global_bound_message(self, variable_name: str, recv_msg: DuctMessage, t: int) -> None:
        """Message handler for GLOBAL_BOUND messages

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: DuctMessage
            received message
        t: int
            message timestamp
        """
        self.logger.debug(f"GLOBAL_BOUND from {variable_name} : {recv_msg.content} at {t}")

        # Store the global upper bound
        self._global_upper_bound = recv_msg.content

        # Flag the normalization as completed
        self.logger.debug(f"Normalization process finished")
        self._normalization_finished = True

        # Send the message to all children
        self._send_message_to_all_children(
            content=self._global_upper_bound, 
            message_type='GLOBAL_BOUND',
        )

    # COST MESSAGE RELATED
    @register('COST')
    def _on_cost_message(self, variable_name: str, recv_msg: DuctMessage, t: int) -> None:
        """Message handler for COST messages.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: DuctMessage
            received message
        t: int
            message timestamp

        """
        self.logger.debug(f"COST from {variable_name} : {recv_msg.content} at {t}")

        # accumulate cost messages until we got the total cost from all our children
        try:
            self._waited_children.remove(variable_name)
        except ValueError as e:
            self.logger.error(
                f"Unexpected COST message from {variable_name} on {self.name} : {recv_msg} "
            )
            raise e

        # process the utility
        cost, bound_value = recv_msg.content
        self._child_costs.append(cost)
        self._child_bounds.append(bound_value)

        # Check if all messages have been received
        if len(self._waited_children) == 0:
            # continue processing the cost messages
            self._process_cost_messages()

    def _process_cost_messages(self) -> None:
        """Process the received cost messages.

        Called when all the cost messages from the children have been received.
        """
        # Calculate the local costs
        self._current_local_cost = self._calculate_local_cost(
            self._current_context, self._current_value,
        )
        # Append the child cost to the local cost
        self._current_cost = self._current_local_cost + np.sum(self._child_costs)
        # Store the total cost as a sample
        self._samples.add_sample(
            self._current_context,
            self._current_value,
            self._current_cost,
        )

        if self._parent_finished:  # line 30
            if self._check_threshold():  # TODO: check functionality line 31
                # Terminate the algorithm lines 32-33
                self._process_termination(self._current_context)
            else:
                # Continue sampling
                self._sample_and_send(self._current_context)
        else:
            if self._check_costs():  # line 37
                # Calculate the bound for the sample
                _, self._optimal_bound = self._generate_duct_sample(self._current_context)
                # Send the cost
                self._send_cost_to_parent(self._current_cost, self._optimal_bound)
            else:
                # Continue sampling
                self._sample_and_send(self._current_context)

    def _check_costs(self) -> bool:
        """Checks the costs (local, current) for feasibility (!= inf), based on line 37 of Algorithm 1"""
        current_cost_check = self._current_cost == np.inf  # y_k^t = inf
        local_cost_checks = []  # exists d l^k(a, d) < inf
        for value in self._variable.domain.values:
            local_cost_check = self._calculate_local_cost(self._current_context, value) < np.inf
            local_cost_checks.append(local_cost_check)

        if current_cost_check and np.any(local_cost_checks):
            return False
        
        return True

    # CONTEXT MESSAGE RELATED
    @register('CONTEXT')
    def _on_context_message(self, variable_name: str, recv_msg: DuctMessage, t: int) -> None:
        """Message handler for CONTEXT messages.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: DuctMessage
            received message
        t: int
            message timestamp
        """
        self.logger.debug(
            f"{self.name}: on context message from {variable_name} : '{recv_msg}' at {t}"
        )
        # Check if the normalization procedure was completed
        # Required for the calculation of the local costs
        assert self._normalization_finished

        # Store the current context
        # Not if you are the root, since the root never has context
        if not self.is_root:
            self._current_context = recv_msg.content

        # Check if the agent is a leaf of the pseudo-tree
        if self.is_leaf:
            # leaf agents can optimize the local costs
            # based on the current context and return
            # the optimal value directly
            optimal_local_cost, _ = self._calculate_optimal_local_cost(self._current_context)
            self._optimal_bound = optimal_local_cost  # as described after Equation 9
            self._send_cost_to_parent(optimal_local_cost, self._optimal_bound)
        else:
            # All other agents need to send a sample and wait for their children
            # to return a COST message based on the CONTEXT
            self._sample_and_send(self._current_context)

    def _calculate_optimal_local_cost(self, context: dict) -> [float, float]:
        """Return the optimal local cost and value
        
        Optimizing over the domain of the (local) variable
        based on the context as defined in line 14 of Algorithm 1:
        l_min = min_d l^k(a, d)
        """
        # Create a local copy of the context 
        local_context = copy.deepcopy(context)
        local_costs = []
        # Calculate the local cost for all the values within the 
        # domain of the local variable
        for value in self._variable.domain.values:
            # Calculate the local cost associated with the local context
            local_cost = self._calculate_local_cost(context, value)
            local_costs.append([local_cost, value])

            # Create the local context based on the context
            # of the parent(s) and the local value
            if local_context is None:
                local_context = {self._variable.name: value}
            else:
                local_context.update({self._variable.name: value})
            self._samples.add_sample(local_context, value, local_cost)
        
        # Get the optimal cost (minimization) and value
        opt_index = np.argmin([c[0] for c in local_costs])
        optimal_local_cost, optimal_value = local_costs[opt_index]

        return optimal_local_cost, optimal_value

    def _get_total_child_costs(self, context: dict) -> None:
        """Send a CONTEXT message to all children and wait for the COST messages"""
        # Create the buffer for the utility messages from the children
        if self._children:
            # Reset the buffers/counters
            self._waited_children = list(self._children)
            self._child_costs = []
            self._child_bounds = []

            # Send the context to all children
            self._send_context_to_children(context)
        else:
            # If the agent does not have any children
            # send the context message to itself
            # such that the flow of the algorithm
            # can remain the same
            # Create the message
            msg = DuctMessage("CONTEXT", context)
            # Send to all children
            self.post_msg(target=self.name, msg=msg)

    def _send_context_to_children(self, context: dict) -> None:
        """Send the CONTEXT message to all children"""
        self._send_message_to_all_children(
            content=context, 
            message_type='CONTEXT',
        )

    def _calculate_local_cost(self, context: dict, sample_value) -> float:
        """Return the normalized local cost based on line 14 from Algorithm 1: l^k(a^t_k, d)"""
        # Create the complete context for the constraints
        local_context = {self._variable.name: sample_value}
        if context is None:
            complete_context = local_context
        else:
            complete_context =  dict(context, **local_context)

        # Check if there is cost for the variable to assign the value
        if hasattr(self._variable, "cost_for_val"):
            total_cost = self._variable.cost_for_val(sample_value)
        else:
            total_cost = 0.0
        
        # Loop over the constraints and sum the costs
        for c in self._constraints:
            assignment_context = {variable_name: complete_context[variable_name] 
                for variable_name in c.scope_names}
            total_cost += c.get_value_for_assignment(assignment_context)
        
        # Apply normalization
        if self._normalization_finished:
            total_cost_normalized = (total_cost - self._local_lower_bound) / self._global_upper_bound
            # Validate the normalization
            assert total_cost_normalized >= 0.0
            assert total_cost_normalized <= 1.0
        else:
            total_cost_normalized = total_cost

        # The algorithm minimizes the sum of the costs
        # Therefore, if the mode is maximization, 
        # the normalized costs can be adjusted
        if self._mode == 'max':
            total_cost_normalized = 1.0 - total_cost_normalized

        return total_cost_normalized
    
    def _check_threshold(self) -> bool:
        """Check the local termination condition based on the threshold (eq. 6)"""
        epsilon_context_values = []
        for value in self._variable.domain.values:
            # tau_{a,d}^{t}
            number_of_counts = self._get_counter_sample(
                self._current_context, 
                value,
            )
            if number_of_counts > 0:
                # CHECK IF THE VALUE IS 'ALLOWED'
                # Calculate the bound
                bound = self._calculate_bound(
                    self._current_context, value)

                # Check if the value is in S_a^t
                opt_cost = self._samples.get_optimal_cost_for_value_under_context(
                    self._current_context, value,
                )
                local_cost = self._calculate_local_cost(
                    self._current_context, value)
                is_possible_sample = bound != local_cost + opt_cost
                
                # Add the value to the list
                if is_possible_sample:
                    # \hat{\mu}_{a}^{t}
                    optimal_cost_context = self._get_optimal_cost_context(self._current_context)
                    # \hat{\mu}_{a, d}^{t}
                    optimal_cost_sample = self._get_optimal_cost_sample(self._current_context, value)

                    # eps = \hat{\mu}_{a}^{t} - (\hat{\mu}_{a, d}^{t} - \sqrt{../..})
                    epsilon_context_value = optimal_cost_context - (
                        optimal_cost_sample - np.sqrt(
                            np.log(2. / self._delta) / number_of_counts
                        )
                    )
                    epsilon_context_values.append(epsilon_context_value)
            else:
                # This value has not been sampled yet, therefore the threshold
                # cannot be reached
                return False

        epsilon_context = np.max(epsilon_context_values)
        inequality_check = epsilon_context <= self._epsilon

        # Return if the threshold has been reached
        if self._parent_finished and inequality_check:
            return True

        return False

    def _get_optimal_cost_context(self, context: dict) -> float:
        """Returns the optimal cost for the context"""
        optimal_cost, _ = self._samples.get_optimal_cost_for_context(context)
        return optimal_cost

    def _get_optimal_cost_sample(self, context: dict, sample_value) -> float:
        """Returns the optimal cost for the sample value given the context"""
        optimal_cost = self._samples.get_optimal_cost_for_value_under_context(context, sample_value)
        return optimal_cost

    def _get_counter_sample(self, context: dict, sample_value) -> int:
        """Returns the amount of times the values has been sampled under the context"""
        count = self._samples.get_count_for_value_under_context(context, sample_value)
        return count

    def _send_cost_to_parent(self, cost: float, bound_value: float) -> None:
        """Sends a COST message to the parent or terminates parent finished"""
        if self.is_root:
            # UPDATE
            # When there is only a single agent
            # the algorithm described in the paper
            # does not function.
            # Therefore, terminate the algorithm
            self._process_termination(self._current_context)
        else:
            # Send optimal message to parent
            if self._parent_finished:
                self._process_termination(self._current_context)
            else:
                self._send_message_to_parent(
                    content=(cost, bound_value),
                    message_type='COST')

    # FINISHED RELATED
    @register('FINISHED')
    def _on_finished_message(self, variable_name: str, recv_msg: DuctMessage, t: int) -> None:
        """Message handler for FINISHED messages.

        Parameters
        ----------
        variable_name: str
            name of the variable that sent the message
        recv_msg: DuctMessage
            received message
        t: int
            message timestamp
        """
        self.logger.debug(
            f"{self.name}: on finished message from {variable_name} : '{recv_msg}' at {t}"
        )

        # Get the context (from the parent)
        self._current_context = recv_msg.content

        # Set the finished flag
        self._parent_finished = True

        # Check the threshold to determine if to terminate 
        # or continue sampling
        if self._check_threshold():
            self._process_termination(self._current_context)
        else:
            self._sample_and_send(self._current_context)

    def _process_termination(self, context: dict) -> None:
        """Terminate the algorithm by setting the value with optimal cost given the context"""
        # Get the optimal local context based on the final message of the parent
        optimal_utility, optimal_local_value = self._samples.get_optimal_cost_for_context(context)
        optimal_local_context = {self._variable.name: optimal_local_value}

        # Append the optimal local context to the parent context
        if context is None:
            final_context = optimal_local_context
        else:
            final_context = dict(context, **optimal_local_context)

        # Send the optimal_context to all children
        self._send_finished_to_children(final_context)

        # Assign the optimal value and finish the algorithm
        self.select_value_and_finish(optimal_local_value, optimal_utility)

    def _send_finished_to_children(self, final_context: dict) -> None:
        """Send the finished message to all children.

        The final context is based on the current context (from ancestors)
        and the optimal (local) value
        """
        self._send_message_to_all_children(
            content=final_context,
            message_type='FINISHED')

