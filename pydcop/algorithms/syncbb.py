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
Implementation of Synchronous Branch and Bound (SBB or SyncBB)

SyncBB simply simulates Branch and Bound in a distributed environment.


Note: Variables and value ordering are given in advance.
* variable are orderer as a simple chain (list)
* values in a domain a ordered

"""
from typing import Optional, List, Any, Tuple

from pydcop.algorithms import ComputationDef
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import assignment_cost, Constraint
from pydcop.infrastructure.computations import (
    VariableComputation,
    SynchronousComputationMixin,
    register,
    message_type,
    ComputationException,
)

GRAPH_TYPE = "ordered_graph"

INFINITY = float("inf")

# Some types definition for the content of messages
VarName = str
VarVal = Any
Cost = float
PathElement = Tuple[VarName, VarVal, Cost]
Path = List[PathElement]

SyncBBForwardMessage = message_type("forward", ["current_path", "ub"])
SyncBBBackwardMessage = message_type("backward", ["current_path", "ub"])
SyncBBTerminateMessage = message_type("terminate", ["current_path", "ub"])

class SyncBBComputation(VariableComputation):
    """

    """

    def __init__(self, computation_definition: ComputationDef):
        super().__init__(computation_definition.node.variable, computation_definition)

        assert computation_definition.algo.algo == "syncbb"
        self.constraints = computation_definition.node.constraints
        self.mode = computation_definition.algo.mode

        node = self.computation_def.node

        self.next_var: VarName = node.get_next()
        self.previous_var: VarName = node.get_previous()
        self.current_path: Path = None
        self.upper_bound = INFINITY if self.mode == "min" else -INFINITY

    def on_start(self):
        # Only done by the first variable in the chain of variables
        if self.previous_var is None:
            path = [(self.variable.name, self.variable.domain[0], 0)]
            ub = INFINITY if self.mode == "min" else -INFINITY
            self.logger.debug(
                f"At startup, first var {self.name} send path {path} to {self.next_var}"
            )
            self.post_msg(self.next_var, SyncBBForwardMessage(path, ub))

    @register("terminate")
    def on_terminate_message(self, sender, msg, t):
        self.logger.debug(
            f"Receiving terminate message at {self.variable.name} from {sender}: "
        )
        if self.next_var is not None:
            self.post_msg(self.next_var, SyncBBTerminateMessage())
        self.finished()


    @register("forward")
    def on_forward_message(self,  sender, recv_msg, t):
        current_path= recv_msg.current_path
        ub= recv_msg.ub

        self.logger.debug(
            f"Receiving forward message at {self.variable.name} from {sender}: "
            f"path: {current_path}, bound: {ub}"
        )
        self.previous_path = current_path

        # Find a new assignment for our variable:
        next_value = get_next_assignment(
            self.variable,
            None,
            self.constraints,
            current_path,
            self.upper_bound,
            mode=self.mode,
        )

        if next_value is None:
            # No possible next value for this variable
            if self.previous_var is None:
                # We are back at the first variable in the ordering, terminate.
                self.post_msg(self.next_var, SyncBBTerminateMessage())
                self.finished()
                self.logger.info(
                    f"Terminate at {self.variable.name}"
                )
            else:
                # Backtrack to previous variable, we cannot select a value for the
                # variable with the partial assignment in current_path
                msg = SyncBBBackwardMessage(self.previous_path, self.upper_bound)
                self.post_msg(self.previous_var, msg)
                self.logger.info(
                    f"Backtracking to {self.previous_var } at {self.variable.name} : "
                    f"no possible value with path {current_path}"
                )

        else:

            if self.next_var is None:
                # Last variable in the chain: the path must be a full assignment.
                # Let's test all values to find the upper bound for this path.
                path_bound = sum(c for _, _, c in current_path)
                value, cost = next_value
                best_val, best_bound = None, self.upper_bound
                while True:
                    if path_bound + cost < best_bound:
                        best_bound = path_bound + cost
                        best_val = value

                    next_value = get_next_assignment(
                        self.variable,
                        value,
                        self.constraints,
                        current_path,
                        self.upper_bound,
                        self.mode,
                    )
                    if next_value is None:
                        break
                    value, cost = next_value
                if best_val is not None:
                    self.upper_bound = best_bound
                    self.value_selection(best_val, self.upper_bound)
                    self.logger.info(
                        f"New upper bound {self.upper_bound} found at {self.variable.name}, "
                        f"with value {best_val}"
                    )
                self.logger.info(
                    f"At end, sending backward from {self.variable.name} "
                    f"to {self.previous_var} with path {current_path}, "
                    f"bound {self.upper_bound}"
                )
                self.post_msg(
                    self.previous_var,
                    SyncBBBackwardMessage(current_path, self.upper_bound),
                )
            else:
                value, cost = next_value
                new_path = current_path.copy()
                new_path.append((self.variable.name, value, cost))
                self.logger.info(
                    f"Value {value} selected at {self.variable.name} "
                    f"sending forward to {self.next_var} "
                    f"with path {new_path}"
                )
                self.post_msg(
                    self.next_var, SyncBBForwardMessage(new_path, self.upper_bound)
                )

    @register("backward")
    def on_backward_msg(self, sender, recv_msg, t):
        current_path = recv_msg.current_path
        self.logger.debug(
            f"Receiving backward message at {self.variable.name} from {sender}: "
            f"path: {current_path}, bound: {recv_msg.ub}"
        )
        var, val, cost = current_path[-1]
        if recv_msg.ub < self.upper_bound:
            self.upper_bound = recv_msg.ub
            self.value_selection(val, self.upper_bound)
        assert var == self.variable.name

        next_val = get_next_assignment(
            self.variable,
            val,
            self.constraints,
            current_path[:-1],
            self.upper_bound,
            self.mode,
        )
        if next_val is not None:
            new_val, new_cost = next_val
            new_path = current_path[:-1]
            new_path.append((self.variable.name, new_val, new_cost))
            self.logger.info(
                f"Trying new value {new_val} at {self.variable.name} when backtracking, "
                f"sending forward to {self.next_var} "
                f"with path {new_path}"
            )
            self.post_msg(
                self.next_var, SyncBBForwardMessage(new_path, self.upper_bound)
            )
        else:
            if self.previous_var is None:
                # First variable in the ordering, and we could not find a possible value
                # No solution !
                self.logger.info(
                    f"Terminate at first variable {self.variable.name}, "
                    f"no next variable when backtracking."
                )
                self.finished()
                self.post_msg(self.next_var, SyncBBTerminateMessage())
            else:
                # Could not find a possible value, backtrack further
                self.logger.info(
                    f"Backtracking at {self.variable.name}: no possible value "
                    f"on backtracking with path {current_path}"
                )
                self.post_msg(
                    self.previous_var,
                    SyncBBBackwardMessage(current_path[:-1], self.upper_bound),
                )


def get_next_assignment(
    variable: Variable,
    current_value: Optional[VarVal],
    constraints: List[Constraint],
    current_path: Path,
    upper_bound: Cost,
    mode: str,
):
    """
    Find the first next value in `variable`'s domain that respects `upper_bound`.

    Parameters
    ----------
    variable: variable
        The variable for which we want to select a value
    current_value: value or `None`
        The value currently assigned to `variable`
    constraints:
        The set of constraints `variable` is involved in
    current_path: Path
        a path assigning a value for each variable before `variable`
    upper_bound: float
        current bound for the path
    mode: str
        "min" or "max"

    Returns
    -------

    """
    candidates = get_value_candidates(variable, current_value)

    found = None
    for candidate in candidates:
        # Check if assigning candidate value to the variable would cause the global
        # cost to exceed the upper-bound.
        candidate_cost = 0
        if not current_path:
            return candidate, 0
        for var, val, elt_cost in current_path:
            var_constraints = constraints_for_variable(constraints, var)
            # This only works for binary constraints, we could extend it to n-ary constraints
            ass_cost = assignment_cost(
                {var: val, variable.name: candidate}, var_constraints
            )
            candidate_cost += ass_cost
            if mode == "min" and (
                candidate_cost >= upper_bound or ass_cost + elt_cost >= upper_bound
            ):
                break  # Try next value in domain.
            elif mode == "max" and (
                candidate_cost <= upper_bound or ass_cost + elt_cost <= upper_bound
            ):
                break  # Try next value in domain.
            else:
                found = candidate, candidate_cost  # Check for next elt in path.
        if found:
            return found

    return None


def constraints_for_variable(
    constraints: List[Constraint], var: VarName
) -> List[Constraint]:
    return [c for c in constraints if var in c.scope_names]


def get_value_candidates(
    variable: Variable, current_value: Optional[VarVal]
) -> List[VarVal]:
    """
    Build an ordered list of candidates values for variable.

    Parameters
    ----------
    variable: Variable
    current_value: value for variable.

    Returns
    -------
    List:
        a list of value that are after `current_value` in the ordered domain of
        `variable`.
    """
    candidates = []
    if current_value is None:
        candidates = list(variable.domain)
    else:
        # Only keep values that are _after_ the current value in the ordered domain:
        reached = False
        for v in variable.domain:
            if reached:
                candidates.append(v)
            if v != current_value:
                continue
            else:
                reached = True
    return candidates
