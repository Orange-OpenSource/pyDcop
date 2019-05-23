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

NCBB: No-Commitment Branch and Bound
------------------------------------

NCBB is a polynomial-space search for DCOP proposed by Chechetka and Sycara in 2006
:cite:`chechetka_no-commitment_2006`.
It is a branch and bound search algorithm with modifications for efficiency, it runs
concurrent search process in different partitions of the search phase.

NCBB defines one computation for each varaible in the DCOP and,
like DPOP, runs on a pseudo-tree, which is automatically built when using the
:ref:`solve<pydcop_commands_solve>` command.

In the current implementation, which maps the original article
:cite:`chechetka_no-commitment_2006` ,
only binary constraints are supported.
According to the authors, it could be extended to n-ary constraints.

NCBB is a synchronous algorithm and is composed of two phases:
* a initialization phase, during which a global upper bound is computed
* a search phase



"""
from typing import Optional, List

from pydcop.algorithms import ComputationDef
from pydcop.infrastructure.computations import (
    VariableComputation,
    SynchronousComputationMixin,
    register,
    message_type,
    ComputationException,
)

GRAPH_TYPE = "pseudotree"


def computation_memory(*args):
    raise NotImplementedError("DPOP has no computation memory implementation (yet)")


def communication_load(*args):
    raise NotImplementedError("DPOP has no communication_load implementation (yet)")


def build_computation(comp_def: ComputationDef):
    return NcbbAlgo(comp_def)


ValueMessage = message_type("ncbb_value", ["value"])
CostMessage = message_type("ncbb_cost", ["cost"])
SearchMessage = message_type("ncbb_search", ["search"])
StopMessage = message_type("ncbb_stop", ["stop"])


class NcbbAlgo(SynchronousComputationMixin, VariableComputation):
    """
    Computation implementation for the NCBB algorithm.

    Parameters
    ----------
    computation_definition: ComputationDef
        the definition of the computation, given as a ComputationDef instance.

    """

    def __init__(self, computation_definition: ComputationDef):
        super().__init__(computation_definition.node.variable, computation_definition)

        assert computation_definition.algo.algo == "ncbb"
        self._mode = computation_definition.algo.mode

        # Set parent, children and constraints
        self._parent = None
        self._children = []
        for l in computation_definition.node.links:
            if l.type == "parent" and l.source == computation_definition.node.name:
                self._parent = l.target
            if l.type == "children" and l.source == computation_definition.node.name:
                self._children.append(l.target)

        # Raise an exception if we pass a non-binary constraint
        self._constraints = []
        for r in computation_definition.node.constraints:
            if r.arity != 2:
                raise ComputationException(
                    f"Invalid constraint {r} with arity {r.arity} "
                    f"for variable {self.name}, "
                    f"NCBB implementation only supports binary constraints."
                )
            self._constraints.append(r)

    @register("ncbb_value")
    def _value_msg_registration(self, variable_name, recv_msg, t):
        pass

    @register("ncbb_cost")
    def _cost_msg_registration(self, variable_name, recv_msg, t):
        pass

    @register("ncbb_search")
    def _search_msg_registration(self, variable_name, recv_msg, t):
        pass

    @register("ncbb_stop")
    def _stop_msg_registration(self, variable_name, recv_msg, t):
        pass

    @property
    def is_root(self):
        return self._parent is None

    @property
    def is_leaf(self):
        return len(self._children) == 0

    def on_start(self):
        # Start with the Initialization phase of NCBB
        # Starting from the root, variable send cost messages down the tree

        pass

    def on_new_cycle(self, messages, cycle_id) -> Optional[List]:

        if not messages:
            return

        msg_types = {msg.type for (sender, msg) in messages.items()}

        if len(msg_types) != 1:
            raise ComputationException()

        msg_type = msg_types.pop()

        if msg_type == "value":
            # Init phase: select a value and send down the tree
            pass
        elif msg_type == "cost":
            pass
        elif msg_type == "search":
            pass
        elif msg_type == "stop":
            pass

        return
