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


import collections
from typing import List, Tuple, Dict, Iterable, Union, Mapping

from pydcop.dcop.objects import AgentDef, Variable, VariableDomain, \
    ExternalVariable, Domain
from pydcop.dcop.relations import RelationProtocol, constraint_from_str, \
    Constraint, filter_assignment_dict


class DCOP(object):
    """A DCOP representation.

    A DCOP is a Constraints Optimization Problem distribution on a set of
    agents: agents send messages to each other to find a solution to the
    optimization problem.
    A DCOP is traditionally represented as a tuple (V, D, C, A, \mu) where A
    is a set of variables, D the set of the domain for these variables, C a set
    of constraints involving these variables, A a set of agents responsible
    for selecting the value of the variable and \mu is a mapping of the
    variable to the agents.
    Given these elements, the goal is to find an assignment of values to
    variables that minimizes the sum of the costs from the constants.

    In pyDcop, a DCOP does not contains the mapping \mu as this mapping
    depends on the algorithm used to solve the constraints optimization.

    """

    def __init__(self, name: str=None,
                 objective: str='min', description: str='',
                 domains: Dict[str, Domain]=None,
                 variables: Dict[str, Variable]=None,
                 constraints: Dict[str, Constraint]=None,
                 agents: Dict[str, AgentDef]=None):
        self.name = name
        self.description = description
        self.objective = objective
        self.domains = {} if domains is None else domains
        self.variables = {} if variables is None else variables
        self._constraints = {} if constraints is None else constraints
        self._agents_def = {} if agents is None else agents
        self.external_variables = {}
        self.dist_hints = None

    @property
    def all_variables(self):
        """
        All variables, internal and external.

        :return: a list of all internal and external variables.
        """
        vs = list(self.variables.values())
        vs.extend(self.external_variables.values())
        return vs

    @property
    def constraints(self):
        return self._constraints

    @property
    def agents(self) -> Dict[str, AgentDef]:
        return self._agents_def

    def agent(self, agt_name: str) -> AgentDef:
        """Return the definition of an agent, given its name.

        Parameters
        ----------
        agt_name: str
            An agent name

        Returns
        -------
        AgentDef
            the definition object for the agent named `agt_name`.

        Raises
        ------
        KeyError
            if there is no agent with this name

        """
        return self._agents_def[agt_name]

    def add_variable(self, v: Variable):
        self.variables[v.name] = v
        return v

    def add_constraint(self, constraint: RelationProtocol):
        """"Add a constraint to te dcop.

        Parameters
        ----------
        constraint : a constraint

        Returns
        -------
        constraint
            the constraint added

        Raises
        ------
        ValueError
            If a domain or variable with the same name, but not the same
            definition, already exist for this dcop.

        Notes
        -----
        Variable(s) and domain(s) involved in the constraint are automatically
        added to the dcop.

        """
        self._constraints[constraint.name] = constraint
        for v in constraint.dimensions:
            current = self.variables.get(v.name, None)
            if current is not None and v != current:
                raise ValueError('Duplicate variable declaration {} from '
                                 'constraint {}'.format(v.name, constraint))
            self.variables[v.name] = v
            self.domains[v.domain.name] = v.domain
        return constraint

    def __add__(self,
                info: Tuple[str, str, List[Variable]]):
        """Convenience notation for adding a constraint to te dcop.

        Parameters
        ----------
        info: a tuple name, expr_str, variables
            name is the name of the constraint, as a string
            expr_str is a python expression as a string, that return the
            value of the constraint. names used in this expression must be
            names of variables given in te third element of the tuple.
            variable can be an iterable of variable object or a dictionary of
            {name: variables}

        Returns
        -------
        DCOP
            the dcop itself, in order to be able to use the += notation.

        Raises
        ------
        ValueError
            If a domain or variable with the same name, but not the same
            definition, already exist for this dcop.

        Notes
        -----
        Variable(s) and domain(s) involved in the constraint are automatically
        added to the dcop.

        See Also
        --------
        add_constraint

        Examples
        --------
        >>> dcop = DCOP('test')
        >>> v1 = Variable('v1', range(10) )
        >>> dcop += 'c1', '2 if v1 > 5 else 10 ', [v1]
        >>> dcop.constraints['c1'](8)
        2
        >>> dcop.constraints['c1'](1)
        10

        """
        (name, expr_str, variables) = info
        if hasattr(variables, 'values'):
            variables = variables.values()

        c = constraint_from_str(name, expr_str, variables)
        self.add_constraint(c)
        return self

    def add_agents(self,
                   agents: Union[AgentDef,
                                 Iterable[AgentDef],
                                 Mapping[str, AgentDef]])-> None:
        """Add agents to the DCOP.

        Agents are given as AgentDef objects.

        Parameters
        ----------
        agents: an AgentDef or an iterable of AgentDef objects of a dict-like
        object associating agents names with AgentDef objects


        Examples
        --------

        The dict-like argument is convenient when creating agents with
        create_agents:

        >>> from pydcop.dcop.objects import create_agents
        >>> dcop = DCOP()
        >>> dcop.add_agents(AgentDef('foo'))
        >>> dcop.add_agents([AgentDef('bar'), AgentDef('tabac')])
        >>> dcop.add_agents(create_agents('a', [1, 2, 3]))
        >>> dcop.agent('a1').name
        'a1'

        """
        if isinstance(agents, AgentDef):
            self._agents_def[agents.name] = agents
        elif isinstance(agents, collections.Mapping):
            self._agents_def.update(agents)
        else:
            for agt in agents:
                self._agents_def[agt.name] = agt

    def domain(self, name: str) -> Domain:
        """Return a domain object, given its name.

        Parameters
        ----------
        name: str
            A Domain name

        Returns
        -------
        AgentDef
            the Domain object

        Raises
        ------
        KeyError
            if there is no domain with this name

        """
        return self.domains[name]

    def variable(self, var_name: str) -> Variable:
        """Return a Variable, given its name

        Parameters
        ----------
        var_name: str
            a variable name

        Returns
        -------
        Variable
            The Variable object

        Raises
        ------
        KeyError
            if there is no variable with this name
        """
        return self.variables[var_name]

    def get_external_variable(self, var_name: str) -> ExternalVariable:
        return self.external_variables[var_name]

    def constraint(self, c_name: str) -> Constraint:
        """Return a Constraint, given its name

        Parameters
        ----------
        c_name: str
            a constraint name

        Returns
        -------
        Constraint
            The Constraint object

        Raises
        ------
        KeyError
            if there is no constraint with this name
        """
        return self.constraints[c_name]

    def solution_cost(self, assignment, infinity):

        # add external variables
        full_assignment = assignment.copy()
        full_assignment.update({v.name: v.value
                                for v in self.external_variables.values()})

        return solution_cost(self.constraints.values(),
                             self.all_variables, full_assignment, infinity)


def solution_cost(relations, variables, assignment, infinity):
    """
    Return the cost of the solution given by the assignment.

    Raises a Value error if the assignment is not a full assignment with a
    value for all variables

    :param relations: a list a relations giving costs
    :param variables: a list a variables objects, only used if there are
    variable with integrated costs
    :param assignment: a dict var_name => value
    :param infinity: A float representing the value used to represent the
    infinity
    """
    cost_hard, cost_soft = 0, 0
    if len(variables) != len(assignment):
        raise ValueError('Cannot compute solution cost : incomplete '
                         'assignment, missing values for vars {}'
                         .format(set(variables) - set(assignment)))

    for r in relations:
        # values = filter_assignment_dict(assignment, r.dimensions)
        # values = {k: v for k, v in values.items() if v is not None}
        # # If we do not yet have a value for all variables, we ignore this
        # # relation in the cost.
        # if len(values) != len(r.dimensions):
        #     raise ValueError('Incomplete assignment')
        # else:
        try:
            r_cost = r(**filter_assignment_dict(assignment, r.dimensions))
        except NameError as ne:
            raise ValueError('Cannot compute solution cost : incomplete '
                             'assignment ' + str(ne))
        # logging.debug('Cost for relation %s : %s ', r.name, r_cost)
        if r_cost != infinity:
            cost_soft += r_cost
        else:
            cost_hard += 1

    for v in variables:

        if v.name in assignment and \
                assignment[v.name] is not None:
            cost_for_val = v.cost_for_val(assignment[v.name])
            if cost_for_val != infinity:
                cost_soft += cost_for_val
            else:
                cost_hard += 1
    return cost_hard, cost_soft


def filter_dcop(dcop: DCOP, accept_unary=False):
    """
    Filters out variables that are not involved in any constraint.

    Returns a new dcop.

    Parameters
    ----------
    dcop: DCOP
        a dcop
    accept_unary: boolean
        if True, keeps variables only involved in binary constraints

    Returns
    -------
    DCOP:
        a dcop that only contains variables involved in at least one
        constraints (non unary, depending on mode)
    """
    variables_alone = set(dcop.variables)

    # find variables that are not involved in any constraint:
    for c in dcop.constraints.values():
        scope = set(v.name for v in c.dimensions)
        if len(scope) == 1:
            if accept_unary:
                variables_alone = variables_alone - set(v.name for v in c.dimensions)
            else:
                continue
        else:
            variables_alone = variables_alone - set(v.name for v in c.dimensions)

    # If we found alone variables, remove their unary constrint if any
    if not accept_unary:
        keep_constraints = {}
        for c_name, c in dcop.constraints.items():
            scope = set(v.name for v in c.dimensions)
            if len(scope) == 1 and scope.issubset(variables_alone):
                continue
            keep_constraints[c_name] = c
    else:
        keep_constraints = dcop.constraints

    variables = {k: v for k, v in dcop.variables.items()
                 if k not in variables_alone}
    filtered = DCOP( dcop.name,
                 dcop.objective, dcop.description,
                 dcop.domains,
                 variables,
                 keep_constraints,
                 dcop.agents)
    return filtered

