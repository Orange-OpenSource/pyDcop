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


import importlib
import types
from numbers import Number
from typing import Callable

from pydcop.utils.various import func_args

"""
Simple Representation module.

This module provide utility methods and mixin to convert python objects
to and from a so called 'simple representation'.

A simple representation is composed only of simple python objects:
* booleans
* string
* numbers
* lists of simple python objects
* dicts of simple python objects
* namedtuple

When using namedtuple, they must obey the two following rules
* be defined at module level (not in class /method)
* the name of the class variable must match the name of the class. for
  example:

    Named = namedtuple('Named', ['foo', 'bar'])


"""


class SimpleReprException(Exception):
    pass


class SimpleRepr(object):
    """
    Mixin to transform python objects into a representation composed only of
    simple python types.
    The idea is that the simple repr can be directly converted into json or
    yaml. The simple representation can be obtained by calling
    `simple_repr(o)`.

    The class using this mixin must satisfy the following constraints:
    * All constructor's parameters must be bool, string, number, objects
      providing a _simple_repr() method (generally by using the the SimpleRep
      mixin) or list or dict of objects of these types.
    * the constructor parameter must map to an attribute with the same name
      preceded by '_'. If it not the case, the class may declare a
      _repr_mapping attribute which maps the argument name(s) with the
      attribute(s) names.


    """
    def _simple_repr(self):

        # Full name = module + qualifiedname (for inner classes)
        r = {'__module__': self.__module__,
             '__qualname__': self.__class__.__qualname__}

        args = [a for a in func_args(self.__init__) if a != 'self']
        for arg in args:
            try:
                val = getattr(self, '_' + arg)
                r[arg] = simple_repr(val)
            except AttributeError:
                if hasattr(self, '_repr_mapping') and arg in \
                        self._repr_mapping:
                    try:
                        r[arg] = self.__getattribute__(
                            self._repr_mapping[arg])
                    except AttributeError:
                        SimpleReprException('Invalid repr_mapping in {}, '
                                            'no attribute for {}'.
                                            format(self,
                                                   self._repr_mapping[arg]))

                else:
                    raise SimpleReprException('Could not build repr for {}, '
                                              'no attribute for {}'.
                                              format(self, arg))
        return r

    @classmethod
    def _from_repr(cls, r):
        """
        This method returns an instance of the class using the mixin,
        built from the simple representation r.

        Most classes do not need to override this class method, but is can
        sometime be useful when dealing with variable arguments for example.

        :param r:
        :return: an instance of the class using the Mixin
        """
        args = {k: from_repr(v) for k, v in r.items()
                if k not in ['__qualname__', '__module__']}
        return cls(**args)


def from_repr(r):
    """
    Build an instance from a simple representation.

    :param r:
    :return:
    """
    if isinstance(r, dict):
        # When we have a dict it can be either a repr for an
        # instance of really a dict!
        if '__qualname__' in r and '__module__' in r:


            if r['__qualname__'] == "tuple":
                # special case for tuple ( not named)
                values = sorted( [(int(i), v) for i, v in r.items()
                                  if i not in ['__qualname__', '__module__']] )
                return tuple([ from_repr(v) for _, v in values])
            module = importlib.import_module(r['__module__'])
            qual = getattr(module, r['__qualname__'])

            if type(qual) == types.FunctionType:
                args = {k: from_repr(v) for k, v in r.items()
                        if k not in ['__qualname__', '__module__', '__type__']}
                M = qual(r['__type__'], args)
                return M(**args)

            if hasattr(qual, '_fields'):
                # special case for namedtuple
                args = {k: from_repr(v) for k, v in r.items()
                        if k not in ['__qualname__', '__module__']}
                return qual.__new__(qual, **args)

            return qual._from_repr(r)
        else:
            return {k: from_repr(v) for k, v in r.items()}
    elif isinstance(r, list):
        return [from_repr(v) for v in r]
    elif isinstance(r, str) or isinstance(r, Number):
        return r


def simple_repr(o):
    """
    Build a simple representation for object o.

    o must already be a simple type (boolean, number, string of list/dict of
    these types) of must implement _simple_repr().

    :param o: an object
    :return: a simple representation for this object
    """
    if hasattr(o, '_simple_repr'):
        return o._simple_repr()
    elif isinstance(o, tuple):
        if hasattr(o, '_asdict'):
            # detect namedtuple
            r = o._asdict()
            r['__module__'] = o.__module__
            r['__qualname__'] = o.__class__.__qualname__
        else:
            r = {i: simple_repr(v) for i, v in enumerate(o)}
            r['__module__'] = o.__class__.__module__
            r['__qualname__'] = o.__class__.__qualname__
        return r
    elif isinstance(o, str) or isinstance(o, Number) or isinstance(o, bool):
        return o
    elif isinstance(o, list) or isinstance(o, tuple) or isinstance(o, set) \
            or isinstance(o, frozenset):
        return [simple_repr(i) for i in o]
    elif isinstance(o, dict):
        return {k: simple_repr(o[k]) for k in o}
    elif o is None:
        return None
    else:
        raise SimpleReprException('Could not build a simple representation '
                                  'for "{}" type={}'.format(o, type(o)))
