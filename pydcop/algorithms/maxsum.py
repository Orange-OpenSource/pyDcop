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
from typing import Optional, List, Dict, Any, Tuple, Union
from collections import defaultdict


from pydcop.algorithms import ComputationDef, AlgoParameterDef
from pydcop.computations_graph.factor_graph import (
    FactorComputationNode,
    VariableComputationNode,
)
from pydcop.dcop.objects import Variable, VariableNoisyCostFunc
from pydcop.dcop.relations import Constraint, generate_assignment_as_dict
from pydcop.infrastructure.computations import (
    DcopComputation,
    SynchronousComputationMixin,
    VariableComputation,
    register,
    Message,
)

GRAPH_TYPE = "factor_graph"
logger = logging.getLogger("pydcop.maxsum")

# Avoid using symbolic infinity as it is currently not correctly
# (de)serialized
# INFINITY = float('inf')
INFINITY = 100000

HEADER_SIZE = 0
UNIT_SIZE = 1

# constants for memory costs and capacity
FACTOR_UNIT_SIZE = 1
VARIABLE_UNIT_SIZE = 1


def build_computation(comp_def: ComputationDef):
    if comp_def.node.type == "VariableComputation":
        logger.debug(f"Building variable computation {comp_def}")
        return VariableComputation(comp_def=comp_def)
    if comp_def.node.type == "FactorComputation":
        logger.debug(f"Building factor computation {comp_def}")
        return FactorComputation(comp_def=comp_def)


def computation_memory(
    computation: Union[FactorComputationNode, VariableComputationNode]
) -> float:
    """Memory footprint associated with the maxsum computation node.

    Notes
    -----
    Two formulations of the memory footprint are possible for factors :
    * If the constraint is given by a function (intentional), the factor
      only needs to keep the costs sent by each variable and the footprint
      is the total size of these cost vectors.
    * If the constraints is given extensively the size of the hypercube of
      costs must also be accounted for.

    Parameters
    ----------
    computation: FactorComputationNode or VariableComputationNode
        A computation node for a factor or a variable in the factor-graph.

    Returns
    -------
    float:
        the memory footprint of the computation.
    """
    if isinstance(computation, FactorComputationNode):
        # Memory footprint associated with the factor computation f.
        # For Maxsum, it depends on the size of the domain of the neighbor
        # variables.
        m = 0
        for v in computation.variables:
            domain_size = len(v.domain)
            m += domain_size * FACTOR_UNIT_SIZE
        return m

    elif isinstance(computation, VariableComputationNode):
        # For Maxsum, the memory footprint a variable computations depends
        #  on the number of  neighbors in the factor graph.
        domain_size = len(computation.variable.domain)
        num_neighbors = len(list(computation.links))
        return num_neighbors * domain_size * VARIABLE_UNIT_SIZE

    raise ValueError(
        "Invalid computation node type {}, maxsum only defines "
        "VariableComputationNodeand FactorComputationNode".format(computation)
    )


def communication_load(
    src: Union[FactorComputationNode, VariableComputationNode], target: str
) -> float:
    """The communication cost of an edge between a variable and a factor.

    Parameters
    ----------
    src: VariableComputationNode
        The ComputationNode for the source variable.
    target: str
        the name of the other variable `src` is sending messages to

    Return
    ------
    float:
        the size of messages between computation and target.
    """
    if isinstance(src, VariableComputationNode):
        d_size = len(src.variable.domain)
        return UNIT_SIZE * d_size + HEADER_SIZE

    elif isinstance(src, FactorComputationNode):
        for v in src.variables:
            if v.name == target:
                d_size = len(v.domain)
                return UNIT_SIZE * d_size + HEADER_SIZE
        raise ValueError(
            "Could not find variable {} in constraint of factor "
            "{}".format(target, src)
        )

    raise ValueError(
        "maxsum communication_load only supports "
        "VariableComputationNode and FactorComputationNode, "
        "invalid computation: " + str(src)
    )


class MaxSumMessage(Message):
    def __init__(self, costs: Dict):
        super().__init__("max_sum", None)
        self._costs = costs

    @property
    def costs(self):
        return self._costs

    @property
    def size(self):
        # Max sum messages are dictionaries from values to costs:
        return len(self._costs) * 2

    def __str__(self):
        return "MaxSumMessage({})".format(self._costs)

    def __repr__(self):
        return "MaxSumMessage({})".format(self._costs)

    def __eq__(self, other):
        if type(other) != MaxSumMessage:
            return False
        if self.costs == other.costs:
            return True
        return False

    def _simple_repr(self):
        r = {"__module__": self.__module__, "__qualname__": self.__class__.__qualname__}

        # When building the simple repr when transform the dict into a pair
        # of list to avoid problem when serializing / de-serializing the repr.
        # The costs dic often contains int as key, when converting to an from
        # json (which only support string for keys in dict), we would
        # otherwise loose the type information and restore the dict with str
        # keys.
        vals, costs = zip(*self._costs.items())
        r["vals"] = vals
        r["costs"] = costs
        return r

    @classmethod
    def _from_repr(cls, r):
        vals = r["vals"]
        costs = r["costs"]

        return MaxSumMessage(dict(zip(vals, costs)))


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
