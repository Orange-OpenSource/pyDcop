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

NCBB defines one computation for each variable in the DCOP and,
like DPOP, runs on a pseudo-tree, which is automatically built when using the
:ref:`solve<pydcop_commands_solve>` command.

In the current implementation, which maps the original article
:cite:`chechetka_no-commitment_2006` ,
only binary constraints are supported.
According to the authors, it could be extended to n-ary constraints.

NCBB is a synchronous algorithm and is composed of two phases:
* a initialization phase, during which a global upper bound is computed
* a search phase

The only other implementation of ncbb I could find is in dcopolis,
however I'm not sure it is correct and it is not fully implemented
(see comments in empty `lowerBound` implementation)
In order to make this algorithm easier to understand, we tried to make the implmentation
as close as possible to the description of the original article.


Initialization phase:

* `VALUE` messages are propagated from top to bottom and a value is selected greedily
  for each variable, starting from the root and based on ancestors values
* The leafs compute an upper bound, which is also propagated upwards in the tree as
`COST` messages
At the end of this phase, each variable has an upper-bound for the sub-tree rooted here.


Search phase:

Main:
* update context
* search
* subtree search
* send stop (1->)

update context:
* agents
  * receive values msg from their ancestors (->2)
    * send lower_bound to their ancestors (3->)
  * receive search msg from their ancestors (->4)
    * update their upper bound
  * receive stop message (->1)

Search:
* start subtree search
* receives costs from children
* send cost to parent

Subtree search:
 * send value to descendants (2->)
 * receive costs from descendants (->3)
 * send search to child, with upper-bound (4->)



"""
from random import choice
from typing import Optional, List

from pydcop.algorithms import ComputationDef
from pydcop.computations_graph.pseudotree import get_dfs_relations
from pydcop.dcop.relations import find_optimal
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


ValueMessage = message_type("value", ["value"])
CostMessage = message_type("cost", ["cost"])
SearchMessage = message_type("search", ["upper_bound"])
SearchValueMessage = message_type("search_value", ["value"])
SearchCostMessage = message_type("search_cost", ["lower_bound"])
StopMessage = message_type("stop", ["stop"])

PHASES = {"INIT", "SEARCH"}


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
        self._parent, self._pseudo_parents, self._children, self._pseudo_children = get_dfs_relations(
            self.computation_def.node
        )
        self.phase = "INIT"
        self._upper_bound = None

        # parent and pseudo-parents:
        self._ancestors = list(self._pseudo_parents)
        self._ancestors.append(self._parent)

        # Children and pseudo-children:
        self._descendants = self._pseudo_children + self._children

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

        self._parents_values = {}
        self._children_costs = {}

    @register("value")
    def _value_msg_registration(self, variable_name, recv_msg, t):
        pass

    @register("cost")
    def _cost_msg_registration(self, variable_name, recv_msg, t):
        pass

    @register("search")
    def _search_msg_registration(self, variable_name, recv_msg, t):
        pass

    @register("search_value")
    def _search_value_msg_registration(self, variable_name, recv_msg, t):
        pass

    @register("search_cost")
    def _search_cost_msg_registration(self, variable_name, recv_msg, t):
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
        if not self.is_root:
            return

        # no cost at root, simply select a value at random and send it to children
        # and pseudo-children
        self.value_selection(choice(self.variable.domain))
        for child in self._descendants:
            self.post_msg(child, ValueMessage(self.current_value))

    def on_new_cycle(self, messages, cycle_id) -> Optional[List]:

        if not messages:
            return

        msg_types = {msg.type for (sender, msg) in messages.items()}

        if len(msg_types) != 1:
            raise ComputationException(
                f"Several types of messages received at {self.name} in the same "
                f"cycle : {msg_types}"
            )

        msg_type = msg_types.pop()

        if msg_type in {"value", "cost"} and self.phase != "INIT":
            raise ComputationException(
                f"{msg_type} messages received at {self.name} while not in INIT phase"
            )

        if (
            msg_type in {"search_value", "search_cost", "search", "stop"}
            and self.phase != "SEARCH"
        ):
            raise ComputationException(
                f"{msg_type} messages received at {self.name} while not in SEARCH phase"
            )

        if msg_type == "value":

            for sender, (message, t) in messages.items():
                self.value_phase(sender, message.value)

        elif msg_type == "cost":

            for sender, (message, t) in messages.items():
                self.cost_phase(sender, message.value)

        elif msg_type == "update_value":
            # Receiving a value message from one of our ancestors
            # FIXMEself._parents_values[sender] = value
            pass

        elif msg_type == "search":
            pass

        elif msg_type == "stop":
            pass

        return

    def value_phase(self, sender, value):

        if sender not in self._ancestors:
            raise ComputationException(
                f"Received at {self.name} value from {sender}, "
                f"which is not an ancestor: {self._ancestors}"
            )

        # Init phase: select a value and send down the tree
        # as value messages are sent by parent and pseudo-parents,
        # they might arrive in several cycles and must be accumulated before we
        # can select our value.
        self._parents_values[sender] = value

        if len(self._parents_values) == len(self._ancestors):
            # Select our own value greedily.
            # For this, we only take into account the constraints with our ancestrors
            ancestors_constraints = []
            for c in self._constraints:
                for v in c.scope_names:
                    if v in self._ancestors:
                        ancestors_constraints.append(c)
                        break
            values, cost = find_optimal(
                self.variable, self._parents_values, ancestors_constraints, self._mode
            )
            self.value_selection(values[0])
            self._upper_bound = cost

            # Send our value to our children.
            if not self.is_leaf:
                for child in self._descendants:
                    self.post_msg(child, ValueMessage(self.current_value))
            else:
                # At leafs, we initiate cost message (sent up) as we don't have to wait
                # for costs from our children.
                for ancestor in self._ancestors:
                    self.post_msg(ancestor, CostMessage(cost))

    def cost_phase(self, sender, cost):
        # compute the upper-bound for the subtree rooted at this variable.

        if sender not in self._children:
            raise ComputationException(
                f"Received cost at {self.name} from {sender}, "
                f"which is not a children: {self._children}"
            )

        self._children_costs[sender] = cost
        self._upper_bound += cost

        if len(self._children_costs) == len(self._children):
            # We have computed the upper-bound for our subtree.
            if not self.is_root:
                # Propagate costs up to parent.
                self.post_msg(self._parent, CostMessage(self._upper_bound))
                # And
            else:
                # If we are the root of the tree, we can now initiate the search phase.
                self.search()

            self.phase = "SEARCH"

    def search(self):
        pass

    def agent_cost(self, assignment):
        pass

    def lower_bound(self, assignment, k):
        # k ancestor index
        pass
