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


from typing import Callable, Tuple, Dict, List

from pydcop.dcop.objects import BinaryVariable
from pydcop.dcop.relations import NAryFunctionRelation, RelationProtocol, \
    Constraint


def create_computation_hosted_constraint(computation_name: str,
                                         bin_vars: Dict[Tuple, BinaryVariable])\
        -> Constraint:
    """
    Create a constraints that the computation names `computation_name` is
    hosted exactly once on a set of n candidate agents.
    The constraints is an hard constraint that return an 'high enough' (10
    000) value when it is not satisfied.

    :param computation_name: the name of the computation
    :param bin_vars:  a dictionary { (comp_name, agt_name) -> BinaryVariable }
     containing the n binary variables x_i^m, one for each of the n
    candidate agents a_m the computation x_i could be hosted on

    :return: a constraint object
    """

    def hosted(**kwargs):
        # kwargs will be a map {bin_var_name -> value} giving the binary
        # value for each of the n candidate agent the computation could be
        # hosted on
        s = sum([v for v in kwargs.values()])
        return 0 if s == 1 else 10000

    constraint = NAryFunctionRelation(
        hosted, list(bin_vars.values()),
        name='{}_hosted'.format(computation_name))

    return constraint


def create_agent_capacity_constraint(agt_name: str, remaining_capacity,
                                     footprint_func: Callable[[str], float],
                                     bin_vars: Dict[Tuple, BinaryVariable]) \
        -> Constraint:
    """
    Create a constraints that ensure that an agent a_m does not exceeds its
    capacity when hosting some candidates computations x_i \in X_c.

    The constraints is an hard constraint that return an 'high enough'
    (10 000) value when it is not satisfied.

    Parameters
    ----------
    agt_name: str
        the name of the agent a_m for which we are creating the constraint
    remaining_capacity
        the remaining capacity of a_m before hosting any candidate computation
        from X_c.
    footprint_func: function
        a function that gives the footprint for a computation, given its name.
    bin_vars: a dictionary { (comp_name, agt_name) -> BinaryVariable }
        containing the k binary variable x_i^m, one for each of the k
        candidate computations x_i that could be hosted on the agent a_m.

    Returns
    -------
    a Constraint object
    """
    # Reversed lookup table: which (agt, computation) a variable is about:
    var_lookup = {v.name: k for k, v in bin_vars.items()}

    def capacity(**kwargs):
        # kwargs will be a name {v_name : value}
        # where v_name is x_{}_{}
        orphaned_footprint = 0
        for v_name in kwargs:
            comp, _ = var_lookup[v_name]
            orphaned_footprint += kwargs[v_name] * footprint_func(comp)

        repair_capa = remaining_capacity - orphaned_footprint
        return 0 if repair_capa >= 0 else 10000

    constraint = NAryFunctionRelation(capacity, list(bin_vars.values()),
                                      name=agt_name + '_capacity')
    return constraint


def create_agent_hosting_constraint(agt_name: str,
                                    hosting_func: Callable[[str], float],
                                    bin_vars: Dict[Tuple, BinaryVariable]) \
        -> Constraint:
    """
    Create a constraints that returns the hosting costs for agent a_m
    `agt_name` when hosting some candidates computations x_i \in X_c.

    The constraints is an soft constraint that should be minimized.

    Parameters
    ----------
    agt_name: str
        the name of the agent a_m.
    hosting_func: Callable
        a function that gives the hosting costs for a computation x_i on a_m,
        given the name of x_i.
    bin_vars: dictionary { (comp_name, agt_name) -> BinaryVariable }
        containing the k binary variable x_i^m, one for each of the k
        candidate computations x_i that could be hosted on the agent a_m.

    Returns
    -------
    a Constraint object
    """

    # Reversed lookup table: which (agt, computation) a variable is about:
    var_lookup = {v.name: k for k, v in bin_vars.items()}

    def hosting_cost(**kwargs):
        cost = 0
        for v_name in kwargs:
            comp, _ = var_lookup[v_name]
            cost += kwargs[v_name] * hosting_func(comp)
        return cost

    constraint = NAryFunctionRelation(hosting_cost, list(bin_vars.values()),
                                      name=agt_name + '_hosting')
    return constraint


def create_agent_comp_comm_constraint(
        agt_name: str, candidate_name: str,
        candidate_info: Tuple[List[str],
                              Dict[str, str],
                              Dict[str, List[str]]],
        comm: Callable[[str, str, str], float],
        bin_vars: Dict[Tuple, BinaryVariable]) \
        -> Constraint:
    """
    Create a constraints that returns the communication costs for agent a_m
    `agt_name` when hosting a candidate computations x_i \in X_c.
    The constraints is an soft constraint that should be minimized.

    Parameters
    ----------
    agt_name: str

    candidate_info:

    comm: Callable
        function (comp_name, neigh_comp, neigh_agt) -> float that returns the
        communication cost between the computation comp_name hosted on the
        current agt_name agent and it's neighbor computation neigh_comp hosted
        on neigh_agt.

    bin_vars: Dict[Tuple, BinaryVariable]
        a dict containing one binary variable for each par (x_i, a_m) where x_i
        is a candidate computation or a neighbor of a candidate computation and
        a_m is an agent that can host x_i.

    Returns
    -------
    A Constraint object
    """
    agts, fixed_neighbors, candidate_neighbors = candidate_info

    def host_cost(**kwargs):
        """

        Parameters
        ----------
        kwargs : dict
            assignment for the variables the comm constraints depends on,
            given as a dict { variable name -> variable value}

        Returns
        -------

        """
        candidate_cost = 0.0
        locally_hosted = bin_vars[(candidate_name, agt_name)].name
        for v in fixed_neighbors:
            v_agt = fixed_neighbors[v]
            candidate_cost += kwargs[locally_hosted] * \
                                comm(candidate_name, v, v_agt)
        for v in candidate_neighbors:
            cost_v = 0.0
            for v_agt in candidate_neighbors[v]:
                arg_name = bin_vars[(v, v_agt)].name
                cost_v += kwargs[arg_name] * comm(candidate_name, v, v_agt)
            candidate_cost += kwargs[locally_hosted] * cost_v

        return candidate_cost

    scope = [bin_vars[(candidate_name, agt_name)]]
    for v in candidate_neighbors:
        for v_agt in candidate_neighbors[v]:
            scope.append(bin_vars[(v, v_agt)])

    constraint = NAryFunctionRelation(
        host_cost, scope, name= 'comm_' + agt_name + '_' + candidate_name)
    return constraint
