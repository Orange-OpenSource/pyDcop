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

The ``pydcop.algorithms`` module  contains the implementation of all DCOP
algorithm supported by pyDCOP.

.. toctree::
  :maxdepth: 1

  algorithms/dba
  algorithms/dsa
  algorithms/adsa
  algorithms/dsatuto
  algorithms/mgm
  algorithms/mgm2
  algorithms/gdba
  algorithms/maxsum
  algorithms/amaxsum
  algorithms/dpop
  algorithms/syncbb


For documentation on how to develop new algorithms,
look at the this
:ref:`tutorial<tutorials_algorithm_implementation>`
and this
:ref:`documentation<implementation_algorithms>`.


``pydcop.algorithms`` also defines objects and functions that are used
to describe and define DCOP algorithms' computations.



.. rubric:: Classes

.. autosummary::

  ComputationDef
  AlgorithmDef
  AlgoParameterDef

.. rubric:: Functions

.. autosummary::

  list_available_algorithms
  load_algorithm_module
  prepare_algo_params
  check_param_value




"""

import inspect
import pkgutil
from functools import lru_cache
from importlib import import_module
from typing import Dict, Any, List, Iterable, NamedTuple, Optional, Union

from pydcop.computations_graph.objects import ComputationNode
from pydcop.utils.simple_repr import SimpleRepr, simple_repr, from_repr

ALGO_STOP = 0
ALGO_CONTINUE = 1
ALGO_NO_STOP_CONDITION = 2


class AlgoParameterDef(NamedTuple):
    """
    Definition of an algorithm's parameter.

    :class:`AlgoParameterDef` instances are used to describe the parameters supported
    by an algorithms.

    For example, dsa supports 3 parameters, which declared with a module-level
    variable in ``dsa.py`` ::

      algo_params = [
        AlgoParameterDef('probability', 'float', None, 0.7),
        AlgoParameterDef('variant', 'str', ['A', 'B', 'C'], 'B'),
        AlgoParameterDef('stop_cycle', 'int', None, 0)]


    """

    name: str
    """
    Name of the parameter (str).
    """
    type: str
    """
    Type of the parameter (str)
    
    This must be either ``int``, ``float`` or ``str`` 
    """

    values: Optional[List[str]] = None
    """
    List of valid values for this parameter.
    
    Can be ``None`` if non-applicable (for a ``float`` paramater, for example).
    """

    default_value: Union[str, int, float] = None
    """
    Default value of the parameter.
    """


class AlgorithmDef(SimpleRepr):
    """
    Full definition of an algorithm's instance.

    An :class:`AlgorithmDef` represents a DCOP algorithm instance
    with all parameters needed to run it.
    These parameters depend on the considered algorithm (e.g.
    variant A, B or C for DSA and damping factor for maxsum)
    and are defined with :class:`AlgoParameterDef`.

    Notes
    -----
    When using the constructor,
    params must already be a dict of valid parameters
    for this algorithm and no validity check is performed.

    Most of the time, you should use
    the :meth:`AlgorithmDef.build_with_default_param`
    static method to create an instance of :class:`AlgorithmDef`,
    as it automatically
    uses default arguments for the requested algorithm.

    Parameters
    ----------
    algo: str
        Name of the algorithm. It must be the name of a module
        in the :mod:`pydcop.algorithms` package.
    params: dict
        Dictionary of algorithm-specific configuration and parameters
    mode: str
        ``'min'`` of ``'max'``, defaults to ``'min'``

    """
    def __init__(self, algo: str, params: Dict[str, Any], mode: str='min') \
            -> None:
        self._algo = algo
        self._mode = mode
        self._params = params

    @staticmethod
    def build_with_default_param(
            algo: str, params: Dict[str, Any]= None,
            mode: str = 'min',
            parameters_definitions: List[AlgoParameterDef]= None):
        """
        Creates an :class:`AlgoDef` instance with defaults parameter values.

        Parameters passed as argument are checked for validity.
        If a value is not provided for some required parameters,
        the default value is used.

        Parameters
        ----------
        algo: str
            Name of the algorithm. It must be the name of a module
            in the :mod:`pydcop.algorithms` package.
        mode: str
            ``'min'`` of ``'max'``, defaults to ``'min'``
        params: dict
            Dictionary of algorithm-specific parameters. If a value is not
            provided for some required parameters, their default value is used.
        parameters_definitions: list of :class:`AlgoParameterDef`
            Algorithms parameters definition. If not provided, their are
            automatically loaded form the algorithm's module.

        Returns
        -------
        algodef: :class:`AlgoDef`
            An instance with defaults parameter values.

        Raises
        ------
        ValueError:
            If an unknown parameter is passed or if a parameter value does not
            respect the parameter definition.

        Examples
        --------

        >>> algo_def = AlgorithmDef.build_with_default_param('dsa', {'variant': 'B'})
        >>> algo_def.param_value('probability')
        0.7
        >>> algo_def.param_value('variant')
        'B'

        """

        if parameters_definitions is None:
            algo_module = load_algorithm_module(algo)
            parameters_definitions = algo_module.algo_params

        params = {} if params is None else params
        params = prepare_algo_params(
            params, parameters_definitions)  # type: Dict[str, Any]

        return AlgorithmDef(algo, params, mode)

    @property
    def algo(self) -> str:
        """
        The name of the algorithm.

        The name of the algorithm is the name of a module
        in the :mod:`pydcop.algorithms` package.

        Returns
        -------
        str: the name of the algorithm.
        """
        return self._algo

    @property
    def mode(self) -> str:
        """
        The mode, ``'min'`` or ``'max'``.

        Returns
        -------
        str: The mode, 'min or 'max'.
        """
        return self._mode

    def param_names(self) -> Iterable[str]:
        """
        Names of the parameters for this algorithm.

        Returns
        -------
        An iterable of str.
        """
        return self._params.keys()

    def param_value(self, param: str) -> Any:
        """
        The value of a parameter.

        Parameters
        ----------
        param: str
            A parameter name

        Returns
        -------
        The value of the parameter

        Raises
        ------
        KeyError: if there is no parameter with this name.
        """
        return self._params[param]

    @property
    def params(self)-> Dict:
        """
        A dictionary of parameters values.

        The dictionary is a copy of the internal parameters and can be safely
        modified.

        Returns
        -------
        A dictionary of parameters values.
        """
        return dict(self._params)

    def _simple_repr(self):
        r = super()._simple_repr()
        r['params'] = simple_repr(self._params)
        return r

    @classmethod
    def _from_repr(cls, r):
        params = r['params']
        del r['params']
        args = {k: from_repr(v) for k, v in r.items()
                if k not in ['__qualname__', '__module__']}
        algo = cls(**args, params=params)
        return algo

    def __str__(self):
        return 'AlgorithmDef({})'.format(self.algo)

    def __repr__(self):
        return 'AlgorithmDef({}, {}, {})'.format(self.algo, self.mode, self._params)

    def __eq__(self, other):
        if type(other) != AlgorithmDef:
            return False
        if self.algo != other.algo or self.mode != other.mode:
            return False
        if self._params != other.params:
            return False
        return True


class ComputationDef(SimpleRepr):
    """
    Full definition of a Computation.

    A Computation node contains all the information needed to create a
    computation instance that can be run. It can be used when deploying the
    computation or as a replica when distributing copies of a computation for
    resilience.

    Parameters
    ----------
    node: ComputationNode
        A computation node
    algo: AlgorithmDef
        algorithm definition ans an `AlgoDef` instance.
    """

    def __init__(self, node: ComputationNode, algo: AlgorithmDef) -> None:
        self._node = node
        self._algo = algo

    @property
    def algo(self) -> 'AlgorithmDef':
        return self._algo

    @property
    def node(self) -> ComputationNode:
        return self._node

    @property
    def name(self):
        return self.node.name

    def __str__(self):
        return 'ComputationDef({}, {})'.format(self.node.name, self.algo.algo)

    def __repr__(self):
        return 'ComputationDef({}, {})'.format(self.node, self.algo)

    def __eq__(self, other):
        if type(other) != ComputationDef:
            return False
        if self.node == other.node and self.algo == other.algo:
            return True
        return False


def check_param_value(param_val: Any, param_def: AlgoParameterDef) -> Any:
    """
    Check if  ``param_val`` is a valid value for a ``AlgoParameterDef``

    When a parameter is given as a str, and the definition expect an int or a
    float, a conversion is automatically attempted and the converted value is
    returned

    Parameters
    ----------
    param_val: any
        a value
    param_def: AlgoParameterDef
        a parameter definition

    Returns
    -------
    param_value:
        the parameter value if it is valid according to the definition (and
        potentially after conversion)


    Raises
    ------
    ValueError:
        Raises a ValueError if the value does not satisfies the parameter
        definition.

    Examples
    --------

    >>> param_def = AlgoParameterDef('p', 'str', ['a', 'b'], 'b')
    >>> check_param_value('b', param_def)
    'b'

    With automatic conversion from str to int
    >>> param_def = AlgoParameterDef('p', 'int', None, None)
    >>> check_param_value('5', param_def)
    5


    """
    if not is_of_type_by_str(param_val, param_def.type):

        if param_def.type == 'int':
            param_val = int(param_val)
        elif param_def.type == 'float':
            param_val = float(param_val)
        else:

            raise ValueError(
                'Invalid type for value {} of parameter {}, must be {}'.format(
                    param_val, param_def.name, param_def.type))

    if param_def.values:
        if param_val in param_def.values:
            return param_val
        else:
            raise ValueError('Invalid value for parameter {}, must be one of '
                             '{}'.format(param_def.name, param_def.values))
    return param_val


def prepare_algo_params(params: Dict[str, Any],
                        parameters_definitions: List[AlgoParameterDef]):
    """
    Ensure algorithm's parameters are valid.

    Check validity of algorithm parameters and add default value for missing
    parameters.

    Parameters
    ----------
    params: Dict[str, Any]
        a dict containing name and values for parameters
    parameters_definitions: list of AlgoParameterDef
        definition of parameters

    Examples
    --------

    >>> param_defs = [AlgoParameterDef('p1', 'str', ['1', '2'], '1'), \
                      AlgoParameterDef('p2', 'int', None, 5),\
                      AlgoParameterDef('p3', 'float', None, 0.5)]
    >>> prepare_algo_params({}, param_defs)['p3']
    0.5
    >>> prepare_algo_params({'p2' : 2}, param_defs)['p2']
    2
    >>> prepare_algo_params({'p3' : 0.7}, param_defs)['p3']
    0.7

    Returns
    -------
    params: dict
        a Dict with all algorithms parameters. If a parameter was not
        provided in the input dict, it is added with its default value.

    Raises
    ------
    ValueError:
        If an unknown parameter is passed or if a parameter value does not
        respect the parameter definition.

    """
    selected_params = {}
    all_algo_params = {param_def.name: param_def
                       for param_def in parameters_definitions}
    for param_name in params:
        if param_name in all_algo_params:
            param_def = all_algo_params[param_name]
            param_val = params[param_name]
            param_val = check_param_value(param_val, param_def)
            selected_params[param_name] = param_val

        else:
            raise ValueError('Unknown parameter for algorithm : {}'
                             .format(param_name))

    missing_params = set(all_algo_params) - set(params)
    for param_name in missing_params:
        selected_params[param_name] = all_algo_params[param_name].default_value

    return selected_params


def list_available_algorithms() -> List[str]:
    """
    The list of available DCOP algorithms.

    Returns
    -------
    a list of str
    """
    exclude_list = {'generic_computations', 'graphs', 'objects'}
    algorithms = []

    root_algo = import_module('pydcop.algorithms')
    for importer, modname, ispkg in pkgutil.iter_modules(root_algo.__path__,
                                                         ''):
        if modname not in exclude_list:
            algorithms.append(modname)

    return algorithms

@lru_cache(maxsize=32)
def load_algorithm_module(algo_name: str):
    """
    Dynamically load an algorithm module.

    This should be used instead of `importlib.import_module` as it adds
    default implementations for some methods, if they have not been defined
    for the algorithm.

    Parameters
    ----------
    algo_name: str
        the name of the algorithm. It must be one of the name returned by
        `list_available_algorithms()`.

    Returns
    -------
    module
        The imported module for this algorithm.
    """

    algo_module = import_module('pydcop.algorithms.'+algo_name)
    algo_module.algorithm_name = algo_name

    if not hasattr(algo_module, 'algo_params'):
        algo_module.algo_params = []

    if not hasattr(algo_module, 'communication_load'):
        algo_module.communication_load = lambda *a, **ka: 1

    if not hasattr(algo_module, 'computation_memory'):
        algo_module.computation_memory = lambda *a, **ka: 1

    if not hasattr(algo_module, 'build_computation'):
        # Injecting the build_computation method will only work
        # with algorithms that defines a single computation type.
        implementations = find_computation_implementation(algo_module)
        algo_module.build_computation = implementations[0]

    return algo_module


def find_computation_implementation(algorithm_module):
    """
    Find VariableComputation subclasses in an algorithm module

    Parameters
    ----------
    algorithm_module

    Returns
    -------
    A list of VariableComputation subclasses.
    """

    from pydcop.infrastructure.computations import VariableComputation
    implementations = []
    for m in inspect.getmembers(algorithm_module, inspect.isclass):
        if m[1] != VariableComputation and \
                issubclass(m[1], VariableComputation):
            implementations.append(m[1])
    return implementations


def is_of_type_by_str(value: Any, type_str: str):
    """
    Check if the type of ``value`` is ``type_str``.

    Parameters
    ----------
    value: any
        a value
    type_str: str
        the expected type of ``value``, given as a str

    Examples
    --------

    >>> is_of_type_by_str(2, 'int')
    True
    >>> is_of_type_by_str("2.5", 'float')
    False

    Returns
    -------
    boolean
    """
    return value.__class__.__name__ == type_str
