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

Synchronous implementation of the MaxSum algorithm



"""
import logging
from typing import Optional, List, Dict, Any
from collections import defaultdict


from pydcop.algorithms import ComputationDef, AlgoParameterDef
from pydcop.algorithms import amaxsum
from pydcop.algorithms.amaxsum import MaxSumMessage
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import Constraint, generate_assignment_as_dict
from pydcop.infrastructure.computations import (
    DcopComputation,
    SynchronousComputationMixin,
    VariableComputation,
    register,
)

GRAPH_TYPE = "factor_graph"
logger = logging.getLogger("pydcop.maxsum")

# Avoid using symbolic infinity as it is currently not correctly
# (de)serialized
# INFINITY = float('inf')
INFINITY = 100000


def build_computation(comp_def: ComputationDef):
    if comp_def.node.type == "VariableComputation":
        logger.debug(f"Building variable computation {comp_def}")
        return VariableComputation(comp_def=comp_def)
    if comp_def.node.type == "FactorComputation":
        logger.debug(f"Building factor computation {comp_def}")
        return FactorComputation(comp_def=comp_def)


# MaxSum and AMaxSum have the same definitions for communication load
# and computation footprints.
computation_memory = amaxsum.computation_memory
communication_load = amaxsum.communication_load

# Some semantic type definition, to make things easier to read and check:
VarName = str
VarVal = Any
Cost = float


class FactorComputation(SynchronousComputationMixin, DcopComputation):
    def __init__(self, comp_def: ComputationDef):
        assert comp_def.algo.algo == "maxsum"
        super().__init__(comp_def.node.factor.name, comp_def)

        self.mode = comp_def.algo.mode
        self.factor = comp_def.node.factor

        # costs : messages for our variables, used to store the content of the
        # messages received from our variables.
        # {v -> {d -> costs} }
        # For each variable, we keep a dict mapping the values for this
        # variable to an associated cost.
        self._costs: Dict[VarName, Dict[VarVal:Cost]] = {}

        # A dict var_name -> (message, count)
        self._prev_messages = defaultdict(lambda: (None, 0))

    @register("max_sum")
    def on_msg(self, variable_name, recv_msg, t):
        # No implementation here, simply used to declare the kind of message supported
        # by this computation
        pass

    def on_new_cycle(self, messages, cycle_id) -> Optional[List]:
        pass


class VariableComputation(SynchronousComputationMixin, VariableComputation):
    def __init__(self, comp_def: ComputationDef):
        super().__init__(comp_def.node.variable, comp_def)
        assert comp_def.algo.algo == "maxsum"

        self.factor_names = [link.factor_node for link in comp_def.node.links]

    @register("max_sum")
    def on_msg(self, variable_name, recv_msg, t):
        # No implementation here, simply used to declare the kind of message supported
        # by this computation
        pass

    def on_start(self) -> None:
        pass

    def on_new_cycle(self, messages, cycle_id) -> Optional[List]:
        pass
