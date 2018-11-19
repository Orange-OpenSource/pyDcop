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


import random
from numbers import Real
from typing import Callable, Sized
from typing import Iterable, Any, Dict, Union, Tuple

import itertools
from typing import List

from pydcop.utils.expressionfunction import ExpressionFunction
from pydcop.utils.simple_repr import SimpleRepr, SimpleReprException

VariableName = str


class Domain(Sized, SimpleRepr, Iterable[Any]):
    """
    A VariableDomain indicates which are the valid values for variables with
    this domain. It also indicates the type of environment state represented
    by there variable : 'luminosity', humidity', etc.

    A domain object can be used like a list of value as it support basic
    list-like operations : 'in', 'len', iterable...
    """

    def __init__(self, name: str, domain_type: str, values: Iterable) -> None:
        """

        :param: name: name of the domain.
        :param domain_type: a string identifying the kind of value in the
                            domain. For example : 'luminosity', 'humidity', ...
        :param values: an array containing the values allowed for the
                       variables with this domain.
        """
        self._name = name
        self._domain_type = domain_type
        self._values = tuple(values)

    @property
    def type(self) -> str:
        return self._domain_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def values(self) -> Iterable:
        return self._values

    def __iter__(self):
        # returns the array
        return self._values.__iter__()

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return len(self._values)

    def __contains__(self, v):
        return v in self._values

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Domain):
            return False
        if self.name == o.name and self.values == o.values and self.type == o.type:
            return True

        return False

    def __str__(self):
        return "VariableDomain({})".format(self.name)

    def __repr__(self):
        return "VariableDomain({}, {}, {})".format(self.name, self.type, self.values)

    def __hash__(self):
        return hash((self._name, self._domain_type, self._values))

    def index(self, val):
        """
        Find the position of a value in the domain

        Parameters
        ----------
        val:
            a value to look for in the domain

        Returns
        -------
        the index of this value in the domain.

        Examples
        --------

        >>> d = Domain('d', 'd', [1, 2, 3])
        >>> d.index(2)
        1

        """
        for i, v in enumerate(self._values):
            if val == v:
                return i
        raise ValueError(str(val) + " is not in the domain " + self._name)

    def to_domain_value(self, val: str):
        """
        Find a domain value with the same str representation

        This is useful when reading value from a file.

        Parameters
        ----------
        val : str
            a string that should match a value in the domain (which may
            contains non-string values, eg int)

        Returns
        -------
        a pair (index, value) where index is the position of the value in the
        domain and value the actual value that matches val.

        Examples
        --------

        >>> d = Domain('d', 'd', [1, 2, 3])
        >>> d.to_domain_value('2')
        (1, 2)

        """
        for i, v in enumerate(self._values):
            if str(v) == val:
                return i, v
        raise ValueError(str(val) + " is not in the domain " + self._name)


# We keep VariableDomain as an alias for the moment, but Domain should be
# preferred.
VariableDomain = Domain

binary_domain = Domain("binary", "binary", [0, 1])


class Variable(SimpleRepr):
    """A DCOP variable.

    This class represents the definition of a variable : a name, a domain
    where the variable can take it's value and an optional initial value. It
    is not used to keep track of the current value assigned to the variable.

    Parameters
    ----------
    name: str
        Name of the variable. You must use a valid python identifier if you
        want to use python expression (given as string) to define
        constraints using this variable.
    domain: Domain or Iterable
        The domain where this variable can take its value. If an iterable
        is given a Domain object is automatically created (named after
        the variable name: `d_<var_name>`.
    initial_value: Any
        The initial value assigned to the variable.

    """

    has_cost = False

    def __init__(
        self, name: str, domain: Union[Domain, Iterable[Any]], initial_value=None
    ) -> None:
        self._name = name
        # If the domain has no name, simply use a named derived from the
        # variable name
        if not hasattr(domain, "__iter__") and not isinstance(domain, VariableDomain):
            raise ValueError(
                "Invalid domain, must be an iterable or " "VariableDomain "
            )
        if not isinstance(domain, Domain):
            domain = Domain("d_" + name, "unkown", domain)
        self._domain = domain
        if initial_value is not None and initial_value not in self.domain.values:
            raise ValueError(
                "Invalid initial value {}, not in domain values"
                " {}".format(initial_value, self.domain.values)
            )
        self._initial_value = initial_value

    @property
    def name(self) -> str:
        return self._name

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def initial_value(self):
        return self._initial_value

    def cost_for_val(self, val) -> float:
        return 0

    def __str__(self):
        return "Variable({})".format(self.name)

    def __repr__(self):
        return "Variable({}, {}, {})".format(self.name, self.initial_value, self.domain)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if (
            self.name == other.name
            and self.initial_value == other.initial_value
            and self.domain == other.domain
        ):
            return True
        return False

    def __hash__(self):
        return hash((self._name, self._domain, self._initial_value))

    def clone(self):
        return Variable(self.name, self.domain, initial_value=self.initial_value)


def create_variables(
    name_prefix: str,
    indexes: Union[str, Tuple, Iterable],
    domain: Domain,
    separator: str = "_",
) -> Dict[Union[str, Tuple[str, ...]], Variable]:
    """Mass creation of variables.

    Parameters
    ----------
    name_prefix: str
        Used as prefix when naming the variables.
    indexes: non-tuple iterable of indexes or tuple of iterables of indexes
        If it not a tuple, a variable is be created for each of
        the index. The index might be a range(see examples).
         If it is a tuple of iterable, a variable is created
        for every possible combinations of values from `indexes`.
    domain: Domain
        The domain for the variables.
    separator: str

    Returns
    -------
    dict
        A dictionary ( index -> variable) where index is a string or a
        tuple of string.

    See Also
    --------
    create_binary_variables

    Examples
    --------
    When passing an iterable of indexes:
    >>> vrs = create_variables('x_', ['a1', 'a2', 'a3'],
    ...                        Domain('color', '', ['R', 'G', 'B']))
    >>> assert isinstance(vrs['x_a2'], Variable)
    >>> assert 'B' in vrs['x_a3'].domain

    When passing a range:
    >>> vrs = create_variables('v', range(10),
    ...                        Domain('color', '', ['R', 'G', 'B']))
    >>> assert isinstance(vrs['v2'], Variable)
    >>> assert 'B' in vrs['v3'].domain


    When passing a tuple of iterables of indexes:
    >>> vrs = create_variables('m_',
    ...                        (['x1', 'x2'],
    ...                         ['a1', 'a2', 'a3']),
    ...                        Domain('color', '', ['R', 'G', 'B']))
    >>> assert isinstance(vrs[('x2', 'a3')], Variable)
    >>> assert vrs[('x2', 'a3')].name == 'm_x2_a3'
    >>> assert 'R' in vrs[('x2', 'a3')].domain

    """
    variables = {}  # type: Dict[Union[str, Tuple[str, ...]], Variable]

    if isinstance(indexes, tuple):
        for combi in itertools.product(*indexes):
            name = name_prefix + separator.join(combi)
            variables[tuple(combi)] = Variable(name, domain)
    elif isinstance(indexes, range):
        digit_count = len(str(indexes.stop - 1))
        for i in indexes:
            name = f"{name_prefix}{i:0{digit_count}d}"
            variables[name] = Variable(name, domain)
    elif hasattr(indexes, "__iter__"):
        for i in indexes:
            name = name_prefix + str(i)
            variables[name] = Variable(name, domain)
    else:
        raise TypeError("indexes must be an iterable or a tuple of iterables")

    return variables


class BinaryVariable(Variable):
    def __init__(self, name: str, initial_value=0) -> None:
        super().__init__(name, binary_domain, initial_value)

    def __str__(self):
        return "BinaryVariable({})".format(self.name)

    def __repr__(self):
        return "BinaryVariable({}, {})".format(self.name, self.initial_value)

    def clone(self):
        return BinaryVariable(self.name, initial_value=self.initial_value)


def create_binary_variables(
    name_prefix: str, indexes, separator: str = "_"
) -> Dict[Union[str, Tuple], BinaryVariable]:
    """Mass creation of binary variables.

    Parameters
    ----------
    name_prefix: str
        Used as prefix when naming the binary variables.
    indexes: non-tuple iterable of indexes or tuple of iterables of indexes
        If it not a tuple, a binary variable is be created for each of
        the index. If it is a tuple of iterable, a binary variable is created
        for every possible combinations of values from `indexes`.
    separator: str

    Returns
    -------
    dict
        A dictionary ( index -> Binary variable) where index is a string or a
        tuple of string.

    See Also
    --------
    create_variables

    Examples
    --------
    When passing an iterable of indexes:
    >>> vrs = create_binary_variables('x_', ['a1', 'a2', 'a3'])
    >>> assert isinstance(vrs['x_a2'], BinaryVariable)


    When passing a tuple of iterables of indexes:
    >>> vrs = create_binary_variables('m_',
    ...                               (['x1', 'x2'],
    ...                                ['a1', 'a2', 'a3']))
    >>> assert isinstance(vrs[('x2', 'a3')], BinaryVariable)
    >>> assert vrs[('x2', 'a3')].name ==  'm_x2_a3'
    >>> vrs = create_binary_variables('m_',
    ...                               (['x1', 'x2'],
    ...                                ['a1', 'a2', 'a3']),
    ...                               separator='B')
    >>> assert vrs[('x2', 'a3')].name == 'm_x2Ba3'

    """
    variables = {}  # type: Dict[Union[str, Tuple[str, ...]], BinaryVariable]

    if isinstance(indexes, tuple):
        for combi in itertools.product(*indexes):
            name = name_prefix + separator.join(combi)
            variables[tuple(combi)] = BinaryVariable(name)
    elif hasattr(indexes, "__iter__"):
        for i in indexes:
            name = name_prefix + str(i)
            variables[name] = BinaryVariable(name)
    else:
        raise TypeError("indexes must be an iterable or a tuple of iterables")

    return variables


class VariableWithCostDict(Variable):
    has_cost = True

    def __init__(
        self,
        name: str,
        domain: Union[VariableDomain, Iterable[Any]],
        costs: Dict[Any, float],
        initial_value=None,
    ) -> None:
        """
        :param name: The name of the variable
        :param domain: A VariableDomain object of a list
        :param costs: a dict that associates a cost for each value in domain
        :param initial_value: optional, if given must be in the domain
        """
        super().__init__(name, domain, initial_value)
        self._costs = costs

    def cost_for_val(self, val) -> float:
        try:
            return self._costs[val]
        except KeyError:
            return 0.0

    def __str__(self):
        return "VariableWithCostDict({})".format(self.name)

    def __repr__(self):
        return "VariableWithCostDict" "({}, {}, {}, {})".format(
            self.name, self.initial_value, self.domain, self._costs
        )

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if (
            self.name == other.name
            and self.initial_value == other.initial_value
            and self.domain == other.domain
            and self._costs == other._costs
        ):
            return True
        return False

    def __hash__(self):
        return super().__hash__() ^ hash(tuple(self._costs.values()))

    def clone(self):
        return VariableWithCostDict(
            self.name, self.domain, self._costs, initial_value=self.initial_value
        )


class VariableWithCostFunc(Variable):
    has_cost = True

    def __init__(
        self,
        name: str,
        domain: Union[VariableDomain, Iterable[Any]],
        cost_func: Union[Callable[..., float], ExpressionFunction],
        initial_value: Any = None,
    ) -> None:
        """
        :param name: The name of the variable
        :param domain: A VariableDomain object of a list
        :param cost_func: a function that returns a cost for each value in the
        domain.
        :param initial_value: optional, if given must be in the domain
        """
        super().__init__(name, domain, initial_value)
        if hasattr(cost_func, "variable_names"):
            # Specific corner case when using an ExpressionFunction as a
            # cost_func: check arguments
            if (
                len(cost_func.variable_names) != 1
                or name not in cost_func.variable_names
            ):
                raise ValueError(
                    "Cost function for var {} must have a single "
                    "variable, which must be the same as "
                    'the variable : "{} != {}'.format(
                        name, name, cost_func.variable_names
                    )
                )
        self._cost_func = cost_func

    def cost_for_val(self, val) -> float:
        if hasattr(self._cost_func, "variable_names"):
            # for function that need keyword arg, like ExpressionFunction
            return self._cost_func(**{self.name: val})
        else:
            return self._cost_func(val)

    def __str__(self):
        return "VariableWithCostFunc({})".format(self.name)

    def __repr__(self):
        return "VariableWithCostFunc" "({}, {}, {}, {})".format(
            self.name, self.initial_value, self.domain, self._cost_func
        )

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if (
            self.name == other.name
            and self.initial_value == other.initial_value
            and self.domain == other.domain
        ):
            if [self.cost_for_val(v) for v in self.domain] == [
                other.cost_for_val(v) for v in other.domain
            ]:
                return True
        return False

    def __hash__(self):
        costs = [self.cost_for_val(v) for v in self.domain]
        return super().__hash__() ^ hash(tuple(costs))

    def clone(self):
        return VariableWithCostFunc(
            self.name, self.domain, self._cost_func, initial_value=self._initial_value
        )

    def _simple_repr(self):
        if not hasattr(self._cost_func, "_simple_repr"):
            raise SimpleReprException(
                "Cannot take a simple repr from a "
                "variable with arbitrary cost function, "
                "use an ExpressionFunction instead"
            )
        else:
            return super()._simple_repr()


class VariableNoisyCostFunc(VariableWithCostFunc):
    has_cost = True

    def __init__(
        self,
        name: str,
        domain: Union[VariableDomain, Iterable[Any]],
        cost_func,
        initial_value=None,
        noise_level: float = 0.02,
    ) -> None:
        """
        :param cost_func: a function that returns a cost for each value in the
        domain.
        """
        super().__init__(name, domain, cost_func, initial_value)

        self._noise_level = noise_level
        self._costs = {}  # type: Dict[Any, float]
        for d in domain:
            self._costs[d] = super().cost_for_val(d) + random.uniform(0, noise_level)

    @property
    def noise_level(self) -> float:
        return self._noise_level

    def cost_for_val(self, val) -> float:
        return self._costs[val]

    def __str__(self):
        return "VariableNoisyCostFunc({})".format(self.name)

    def __repr__(self):
        return "VariableNoisyCostFunc" "({}, {}, {}, {}, {})".format(
            self.name,
            self.initial_value,
            self.domain,
            self._cost_func,
            self._noise_level,
        )

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if (
            self.name == other.name
            and self.noise_level == other.noise_level
            and self.domain == other.domain
            and self._cost_func == other._cost_func
            and self.initial_value == other.initial_value
        ):
            return True
        return False

    def __hash__(self):
        # hash on costs without noise
        costs = [
            super(VariableNoisyCostFunc, self).cost_for_val(d) for d in self.domain
        ]
        return Variable.__hash__(self) ^ hash(tuple(costs))

    def clone(self):
        return VariableNoisyCostFunc(
            self.name,
            self.domain,
            self._cost_func,
            initial_value=self.initial_value,
            noise_level=self._noise_level,
        )


class ExternalVariable(Variable):
    """
    An external is a variable that is not subject to optimization: its value
    cannot be changed by DCOP algorithms, which only use it as an input,
    read-only, parameter.
    The value of an external variable can still change for external reasons,
    in that case computation(s) should adapt to the change when appropriate. 
    One can be notified of such change by subscribing to the ExternalVariable.

    External variable can be used to represent the value from a sensor for
    example. : it can actually be changed to match the value read from a real
    sensor or manually by the user (when using a simulator).
    """

    def __init__(
        self, name: str, domain: Union[VariableDomain, Iterable[Any]], value=None
    ) -> None:
        super().__init__(name, domain)
        self._cb = []  # type: List[Callable[[Any], Any]]
        self._value = list(domain.values)[0]
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if val == self._value:
            return
        if val not in self._domain:
            raise ValueError(
                "Invalid value {} for sensor variable {}".format(val, self._name)
            )
        self._value = val
        self._fire(val)

    def subscribe(self, callback):
        self._cb.append(callback)

    def unsubscribe(self, callback):
        self._cb.remove(callback)

    def _fire(self, value):
        for cb in self._cb:
            cb(value)

    def clone(self):
        return ExternalVariable(self.name, self.domain, self.value)


class AgentDef(SimpleRepr):
    """Definition of an agent.

    AgentDef objects are used when only the definition of the agent is needed,
    and not the actual running agents. This is for example the case when
    computing the computations' distribution, or when instanciating concrete
    agents.

    Notes
    -----
    Route cost default to 1 because they are typically used as a multiplier
    for message cost when calculating communication cost.
    On the other hand, hosting cost default to 0 because they are used in a
    sum.
    In order to allow using problem-specific attribute on agents, any named
    argument passed when creating an AgentDef is available as an attribute

    Examples
    --------

    >>> a1 = AgentDef('a1', foo='bar')
    >>> a1.name
    'a1'
    >>> a1.foo
    'bar'


    Parameters
    ----------
    name: str
        the name of the agent
    default_route: float
        the default cost of a route when not specified in routes.
    routes: dictionary of agents name, as string, to float
        attribute a specific route cost between this agent and the agents
        whose names are used as key in the dictionary
    default_hosting_cost
        the default hosting for a computation when not specified in
        hosting_costs.
    hosting_costs: dictionary of computation name, as string, to float
        attribute a specific cost for hosting the computations
        whose names are used as key in the dictionary.
    kwargs: dictionary string -> any
        any extra attribute that should be available on this AgentDef
        object.

    """

    def __init__(
        self,
        name: str,
        default_route: float = 1,
        routes: Dict[str, float] = None,
        default_hosting_cost: float = 0,
        hosting_costs: Dict[str, float] = None,
        **kwargs: Union[str, int, float],
    ) -> None:
        """Build an AgentDef, only the name is mandatory."""
        super().__init__()
        self._name = name
        self._attr = kwargs
        self._default_hosting_cost = default_hosting_cost
        self._hosting_costs = hosting_costs if hosting_costs is not None else {}
        self._default_route = default_route
        self._routes = routes if routes is not None else {}

    @property
    def name(self) -> str:
        return self._name

    def hosting_cost(self, computation: str) -> float:
        """The cost for hosting a computation.

        Parameters
        ----------
        computation: str
            the name of the computation

        Returns
        -------
        float
            the cost for hosting a computation

        Examples
        --------
        >>> agt = AgentDef('a1', default_hosting_cost=3)
        >>> agt.hosting_cost('c2')
        3
        >>> agt.hosting_cost('c3')
        3

        >>> agt = AgentDef('a1', hosting_costs={'c2': 6})
        >>> agt.hosting_cost('c2')
        6
        >>> agt.hosting_cost('c3')
        0

        """
        try:
            return self._hosting_costs[computation]
        except KeyError:
            return self.default_hosting_cost

    @property
    def default_hosting_cost(self) -> float:
        return self._default_hosting_cost

    @property
    def hosting_costs(self) -> Dict[str, float]:
        return self._hosting_costs

    @property
    def default_route(self) -> float:
        return self._default_route

    @property
    def routes(self) -> Dict[str, float]:
        return self._routes

    def route(self, other_agt: str) -> float:
        """The route cost between this agent and other_agent.

        Parameters
        ----------
        other_agt: str
            the name of the other agent

        Returns
        -------
        float
            the cost of the route

        Examples
        --------
        >>> agt = AgentDef('a1', default_route=5)
        >>> agt.route('a2')
        5
        >>> agt.route('a1')
        0

        >>> agt = AgentDef('a1', routes={'a2':8})
        >>> agt.route('a2')
        8
        >>> agt.route('a3')
        1
        """
        if self.name == other_agt:
            return 0
        try:
            return self._routes[other_agt]
        except KeyError:
            return self.default_route

    def extra_attr(self) -> Dict[str, Any]:
        """
        Extra attributes for this agent definition.

        These extra attributes are the `kwargs` passed to the constructor.
        They are typically used to defined extra properties on an agent,
        like the capacity.

        Returns
        -------
        Dictionary of strings to values,
        """
        if self._attr is None:
            return dict()
        return self._attr

    def __getattr__(self, item):
        try:
            return self._attr[item]
        except KeyError:
            raise AttributeError("No attribute " + str(item) + " on " + str(self))

    # When using the process mode, AgentDef objects are pickled to be
    # passed to another process. because we use the special method
    # __getattr__, we must provide a __getstate__ and __setstate__ method
    # for pickle support.

    def __getstate__(self):
        return (self._name, self._hosting_costs, self.default_hosting_cost, self._attr)

    def __setstate__(self, state):
        (
            self._name,
            self._hosting_costs,
            self._default_hosting_cost,
            self._attr,
        ) = state

    def __str__(self):
        return "AgentDef({})".format(self.name)

    def __repr__(self):
        return "AgentDef({}, {})".format(self.name, self._attr)

    def __eq__(self, other):
        if type(other) != AgentDef:
            return False
        if (
            self.name == other.name
            and self.hosting_costs == other.hosting_costs
            and self._attr == other._attr
            and self.default_hosting_cost == other.default_hosting_cost
        ):
            return True
        return False


def create_agents(
    name_prefix: str,
    indexes: Union[Iterable, Tuple[Iterable]],
    default_route: float = 1,
    routes: Dict[str, float] = None,
    default_hosting_costs: float = 0,
    hosting_costs: Dict[str, float] = None,
    separator: str = "_",
    **kwargs: Union[str, int, float],
) -> Dict[Union[str, Tuple[str, ...]], AgentDef]:
    """Mass creation of agents definitions.

    Parameters
    ----------
    name_prefix: str
        Used as prefix when naming the agents.
    indexes: non-tuple iterable of indexes or tuple of iterable of indexes
        If it not a tuple, an AgentDef is be created for each of
        the index. If it is a tuple of iterable, an AgentDef is created
        for every possible combinations of values from `indexes`.
    default_route: float
        The default cost of a route when not specified in routes.
    routes: dictionary of agents name, as string, to float
        Attribute a specific route cost between this agent and the agents
        whose names are used as key in the dictionary
    default_hosting_costs
        The default hosting for a computation when not specified in
        hosting_costs.
    hosting_costs: dictionary of computation name, as string, to float
        Attribute a specific cost for hosting the computations
        whose names are used as key in the dictionary.
    separator: str
    kwargs: dictionary

    Returns
    -------
    dict
        A dictionary ( index -> AgentDef) where index is a string or a
        tuple of string.

    See Also
    --------
    create_variables

    Examples
    --------
    When passing an iterable of indexes:
    >>> agts = create_agents('a', ['1', '2', '3'],
    ...                      default_route=2, default_hosting_costs=7)
    >>> assert isinstance(agts['a2'], AgentDef)

    When passing a range:
    >>> agts = create_agents('a', range(20),
    ...                      default_route=2, default_hosting_costs=7)
    >>> assert isinstance(agts['a08'], AgentDef)

    """
    agents = {}  # type: Dict[Union[str, Tuple[str, ...]], AgentDef]

    if isinstance(indexes, tuple):
        for combi in itertools.product(*indexes):
            name = name_prefix + separator.join(combi)
            agents[tuple(combi)] = AgentDef(
                name,
                default_route=default_route,
                routes=routes,
                default_hosting_costs=default_hosting_costs,
                hosting_costs=hosting_costs,
                **kwargs,
            )
    elif isinstance(indexes, range):
        digit_count = len(str(indexes.stop - 1))
        for i in indexes:
            name = f"{name_prefix}{i:0{digit_count}d}"
            agents[name] = AgentDef(
                name,
                default_route=default_route,
                routes=routes,
                default_hosting_costs=default_hosting_costs,
                hosting_costs=hosting_costs,
                **kwargs,
            )
    elif hasattr(indexes, "__iter__"):
        for i in indexes:
            name = name_prefix + str(i)
            agents[name] = AgentDef(
                name,
                default_route=default_route,
                routes=routes,
                default_hosting_costs=default_hosting_costs,
                hosting_costs=hosting_costs,
                **kwargs,
            )
    else:
        raise TypeError("indexes must be an iterable or a tuple of iterables")

    return agents
