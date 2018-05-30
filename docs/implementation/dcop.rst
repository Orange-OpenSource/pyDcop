
.. _implementation_dcop:

DCOP objects
============

Objects used for representing DCOP are in the package `pydcop.dcop`

VariableDomain:

* Domain: represented by pydcop.dcop.objects.VariableDomain
* Variable
  several kind of variables
  domain
  simple list of value can also be used as a domain

* Constraints (aka Relation)

  relation_from_str(espression, variables)

        d1 = VariableDomain('d1', '', [1, 2, 3, 5])
        v1 = Variable('v1', d1)
        f1 = relation_from_str('f1', 'v1 * 0.5', [v1])

* Agents
