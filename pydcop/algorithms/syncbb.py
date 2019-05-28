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

INFINITY = float("inf")

# Some types definition for the content of messages
VarName = str
VarVal = Any
Cost = float
PathElement = Tuple[VarName, VarVal, Cost]
Path = List[PathElement]

SyncBBForwardMessage = message_type("forward", ["current_path", "ub"])
SyncBBBackwardMessage = message_type("backward", ["current_path", "ub"])


class SynBBComputation(SynchronousComputationMixin, VariableComputation):
    """

    """

    def __init__(self, computation_definition: ComputationDef):
        super().__init__(computation_definition.node.variable, computation_definition)

        assert computation_definition.algo.algo == "syncbb"
        self.constraints = computation_definition.node.constraints
        self.mode = computation_definition.algo.mode

        # TODO: define a graph model for simple chain or variable / ordering
        self.next_var: VarName = None
        self.previous_var: VarName = None
        self.current_path: Path = None
        self.upper_bound = INFINITY if self.mode == "min" else -INFINITY

    def on_start(self):
        # Only done by the first variable in the chain of variables
        if self.previous is None:
            path = [(self.variable.name, self.variable.domain[0], 0)]
            ub = INFINITY if self.mode == "min" else -INFINITY
            msg = SyncBBForwardMessage(path, ub)
            self.post_msg(self.next, msg)

    @register("forward")
    def on_forward_msg(self, variable_name, recv_msg, t):
        pass

    @register("backward")
    def on_backward_msg(self, variable_name, recv_msg, t):
        pass

    def on_new_cycle(self, messages, cycle_id) -> Optional[List]:

        if len(messages) > 1:
            raise ComputationException(
                f"Received {len(messages)} in a single cycle at {self.name}, "
                f"while SyncBB at at most one message per cycle"
            )
        if not messages:
            return
        message = messages[0]

        if message.type == "forward":
            self.on_forward_message(message.current_path, message.ub)
        elif message.type == "backward":
            self.on_backward_message(message.current_path, message.ub)

    def on_forward_message(self, current_path: Path, ub: Cost):
        pass

    def on_backward_message(self, current_path: Path, ub: Cost):
        pass

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
