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


import unittest
from collections import namedtuple

from pydcop.utils.simple_repr import SimpleRepr, SimpleReprException, \
    simple_repr, from_repr


class A(SimpleRepr):
    def __init__(self, attr1, attr2):
        self._attr1 = attr1
        self._attr2 = attr2


class TestAttrHaveSameNameAsInitParams(unittest.TestCase):

    """
    Tests for the case where all attributes in the class maps directly to
    argument of the __init__ constructor:

    e.g.

    A.__init__(a) --> A._a
    """

    def test_simple_attr_only(self):

        a = A('foo', 'bar')
        r = a._simple_repr()

        self.assertEqual(r['attr1'], 'foo')
        self.assertEqual(r['attr2'], 'bar')

    def test_simple_attr_only_with_bool(self):

        a = A(False, True)
        r = a._simple_repr()

        self.assertEqual(r['attr1'], False)
        self.assertEqual(r['attr2'], True)

    def test_simple_attr_only_with_None(self):

        a = A(False, None)
        r = a._simple_repr()

        self.assertEqual(r['attr1'], False)
        self.assertEqual(r['attr2'], None)

    def test_from_repr_simple_attr_only(self):

        a = A('foo', 'bar')
        r = a._simple_repr()

        b = from_repr(r)
        self.assertTrue(isinstance(b, A))
        self.assertEqual(b._attr1, 'foo')
        self.assertEqual(b._attr2, 'bar')

    def test_list_attr(self):

        a = A('foo', [1, 2, 3])
        r = a._simple_repr()
        self.assertEqual(r['attr1'], 'foo')
        self.assertEqual(r['attr2'], [1, 2, 3])

    def test_from_repr_list_attr(self):

        a = A('foo', [1, 2, 3])
        r = a._simple_repr()

        b = from_repr(r)
        self.assertEqual(b._attr1, a._attr1)
        self.assertEqual(b._attr2, a._attr2)

    def test_dict_attr(self):

        a = A('foo', {'a': 1, 'b': 2})
        r = a._simple_repr()
        self.assertEqual(r['attr1'], 'foo')
        self.assertEqual(r['attr2'], {'a': 1, 'b': 2})

    def test_from_repr_dist_attr(self):

        a = A('foo', {'a': 1, 'b': 2})
        r = a._simple_repr()

        b = from_repr(r)
        self.assertEqual(b._attr1, a._attr1)
        self.assertEqual(b._attr2, a._attr2)

    def test_object_attr(self):

        a2 = A('foo2', 'bar2')
        a = A('foo', a2)
        r = a._simple_repr()

        self.assertEqual(r['attr1'], 'foo')
        self.assertEqual(r['attr2'], a2._simple_repr())
        self.assertEqual(r['attr2']['attr1'], 'foo2')

    def test_from_repr_object_attr(self):

        a2 = A('foo2', 'bar2')
        a = A('foo', a2)
        r = a._simple_repr()

        b = from_repr(r)
        self.assertIsInstance(b, A)
        self.assertEqual(b._attr1, 'foo')
        self.assertEqual(b._attr2._attr1, 'foo2')
        self.assertEqual(b._attr2._attr2, 'bar2')

    def test_list_of_objects(self):

        a2 = A('foo2', 'bar2')
        a3 = A('foo3', 'bar3')
        a = A('foo', [a2, a3])
        r = a._simple_repr()
        self.assertEqual(r['attr1'], 'foo')
        self.assertTrue(isinstance(r['attr2'], list))
        self.assertEqual(r['attr2'][0], a2._simple_repr())
        self.assertEqual(r['attr2'][1]['attr1'], 'foo3')

    def test_from_repr_list_of_objects(self):

        a2 = A('foo2', 'bar2')
        a3 = A('foo3', 'bar3')
        a = A('foo', [a2, a3])
        r = a._simple_repr()

        b = from_repr(r)
        self.assertIsInstance(b, A)
        self.assertEqual(b._attr1, 'foo')
        self.assertTrue(isinstance(b._attr2, list))
        self.assertEqual(b._attr2[0]._attr1, 'foo2')
        self.assertEqual(b._attr2[1]._attr2, 'bar3')

    def test_dict_of_objects(self):

        a2 = A('foo2', 'bar2')
        a3 = A('foo3', 'bar3')
        a = A('foo', {'a': a2, 'b': a3})
        r = a._simple_repr()
        self.assertEqual(r['attr1'], 'foo')
        self.assertTrue(isinstance(r['attr2'], dict))
        self.assertEqual(r['attr2']['a'], a2._simple_repr())
        self.assertEqual(r['attr2']['b']['attr1'], 'foo3')

    def test_composite_list_dict(self):
        a2 = A('foo2', 'bar2')
        a = A('foo', ['a', {'k1': 1, 'k2': a2}, 3])
        r = a._simple_repr()
        self.assertEqual(r['attr1'], 'foo')
        self.assertTrue(isinstance(r['attr2'], list))
        self.assertTrue(isinstance(r['attr2'][1], dict))
        self.assertEqual(r['attr2'][1]['k2']['attr2'], 'bar2')

    def test_raise_when_object_does_not_use_mixin(self):

        class NoMixin(object):

            def __init__(self, a1):
                self.foo = a1

        o = NoMixin('bar')
        self.assertRaises(SimpleReprException, simple_repr, o)

    def test_raise_when_no_corresponding_attribute(self):

        class NoCorrespondingAttr(SimpleRepr):

            def __init__(self, a1):
                self.foo = a1

        o = NoCorrespondingAttr('bar')
        self.assertRaises(SimpleReprException, simple_repr, o)

    def test_mapping_for_corresponding_attribute(self):

        class MappingAttr(SimpleRepr):

            def __init__(self, a1):
                self._repr_mapping = {'a1': 'foo'}
                self.foo = a1

        o = MappingAttr('bar')
        r = simple_repr(o)
        self.assertEqual(r['a1'], 'bar')

    def test_tuple_simple_repr(self):

        a1 = A('foo', ('b', 'a'))
        r = simple_repr(a1)
        print(r)

    def test_tuple_from_repr(self):

        a1 = A('foo', ('b', 'a'))
        r = simple_repr(a1)
        a2 = from_repr(r)
        print(a2)

    def test_namedtuple(self):

        # Named = namedtuple('Named', ['foo', 'bar'])
        n = Named(1, 2)
        r = simple_repr(n)

        self.assertEqual(r['foo'], 1)
        self.assertEqual(r['bar'], 2)

        obtained = from_repr(r)

        self.assertEqual(obtained, n)

    def test_namedtuple_complex(self):

        # Named = namedtuple('Named', ['foo', 'bar'])
        n = Named({'a': 1, 'b': 2}, [1, 2, 3, 5])
        r = simple_repr(n)

        self.assertEqual(r['foo'], {'a': 1, 'b': 2})
        self.assertEqual(r['bar'], [1, 2, 3, 5])

        obtained = from_repr(r)

        self.assertEqual(obtained, n)


Named = namedtuple('Named', ['foo', 'bar'])

# TODO :
# optional parameter
# parameters that does map to an attribute
#  attribute with different name ?
#  function needed to compute the value of the attribute
# parameter that should be ignored ?
# from_repr : instanciate an object from the repr

# Include the class name in the repr ?
