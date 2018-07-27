

.. _implementation_algorithms:

DCOP algorithm Implementation
=============================

**Note:** This document build upon the tutorial
:ref:`tutorials_algorithm_implementation`,
you should follow it before reading this.


By providing all the infrastructure, pyDCOP makes it easier to implement a
new DCOP algorithm ; you only have one python module to implement,
with only one mandatory class.


To implement an algorithm you must:

- create a python module
- define one or several :class:`.Message`
- implement your logic in one or several
  :class:`.DcopComputation` class

Optionally, you may also

- declare the algorithm's parameters
- implement some other methods used for some
  distribution methods


Module
------

An algorithm must be defined in it's own module in :py:mod:`pydcop.algorithms`.
For example, ``dsa`` is implemented in the module :py:mod:`pydcop.algorithms.dsa`.
The name of the module is the name you will use
when running your algorithm with the ``pydcop``
command line interface (``-a`` or ``--algo`` parameter).
For example, for a new algorithm named ``my_algorithm``,
you simply create a file ``my_algorithm.py`` in ``pydcop/algorithms``.
You will then be able to use this algorithm
to :ref:`solve<pydcop_commands_solve>` a DCOP with the following command::

  pydcop solve --algo my_algorithm [...]


The module of your algorithm **must** also have an attribute
named ``GRAPH_TYPE`` which contains the name of the computation graph type used.
Available computation graph types are ``'factor_graph'``, ``'pseudo_tree'`` and
``'constraints_hypergraph'``, other could be defined in the future.

For example, in ``dsa.py``::

    GRAPH_TYPE = 'constraints_hypergraph'


Messages
--------

DCOP algorithms are *message passing* algorithms: they work by sending
messages to each other. You must define the message(s) used by your algorithm.
The easiest approach is to use the
:py:func:`.message_type`
class factory method to define your message(s).

For example, the following will define a message type ``MyMessage``, with two
fields ``foo`` and ``bar``::

  MyMessage = message_type('MyMessage', ['foo', 'bar'])

You can then use ``MyMessage`` like any class::

  >>> msg = MyMessage(foo=42, bar=21)
  >>> msg.foo
  42
  >>> msg.type
  'MyMessage'

You can also subclass :py:class:`pydcop.infrastructure.computations.Message`,
which is more verbose but can be convenient if you want to use python's type
annotations::

  class MyMessage(Message):
      def __init__(self, foot: int, bar; float):
      super().__init__('my_message', None)
      self._foo = foo
      self._bar = bar

      @property
      def foo(self) -> int:
          return self._foo

      @property
      def bar(self) -> float:
          return self._bar

In any case, your messages **must** use the
:py:class:`pydcop.utils.simple_repr.SimpleRepr` mixin
(:py:mod:`.Message` already extends it)
for your message to be serializable.
When subclassing :class:`.Message` or
using :py:func:`.message_type` this is done automatically.
This is necessary when running the agents in
different processes, as messages will be sent over the network.


Computation
-----------

An algorithms consists in one or several :py:class:`DcopComputation` class.
Most algorithms have one single type of computation, which is
responsible for selecting the value for a single variable.
In this case you must subclass :py:class:`VariableComputation`,
which provides some convenient methods for value selection.

For more complex algorithm, you can define several computations
(with pyDCOP, your algorithm can have as many kind of computation as you want),
look at MaxSum's implementation for an example
(`MaxSum` has two kind of computations, for `Factor` and `Variable`).


Receiving messages
^^^^^^^^^^^^^^^^^^

At runtime, an instance of a computation is deployed on an agent,
which notifies it when receiving a message.
The computation then processes the message and,
if necessary, emits new messages for other computations.

For each message type, you must declare a handler method using the
:func:`register` decorator::

  @register("my_message_type")
  def _on_my_message(self, sender_name, msg, t):
      # handle message of type 'my_message'
      # sender_name is the name of the computation that sent the message
      # t is the time the message was received by the agent.


Sending messages
^^^^^^^^^^^^^^^^

When sending messages, a computation never needs
to care about the agent hosting the target computations :
all message routing and delivery is taken care of by
the agent and communication infrastructure.
Messages are sent by calling ``self.post_msg``::

  self.post_msg(target_computation_name, message_object)

You can also send a message to all neighbors by using
``self.post_to_all_neighbors``.

Selecting a value
^^^^^^^^^^^^^^^^^

In your computation, when selecting a value for a variable, you **must**
call ``self.value_selection`` with the value and the associated local cost.
This is allows pyDcop to monitor value selection on each agent and
extract the final assignment::

    self.value_selection(self._v.initial_value, local_cost)

The ``local_cost`` is the cost as seen from this variable.

Cycles
^^^^^^

Each your algorithm has a concept of cycle
(i.e. it works in sycnhronized steps), you should call
``self.new_cycle()`` when you start a new cycle.


Terminating the algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^

TODO

Argument Parameters
^^^^^^^^^^^^^^^^^^^
If the algorithm supports parameters, you must give a definition of these
parameters in your module, by defining a variable named ``algo_params``
that contains a list of :class:`AlgoParameterDef`.

See for example in mgm implementation::

    algo_params = [
        AlgoParameterDef('break_mode', 'str', ['lexic', 'random'], 'lexic'),
        AlgoParameterDef('stop_cycle', 'int', None, None),
    ]


These definitions will be automatically used
(with :py:func:`pydcop.algorithms.prepare_algo_params`) to check parameters
for validity and add default values.

An ``AlgoritmDef`` instance populated with the parsed parameters will be passed to
your ``__init__`` method, you can then use it to pass these parameters
to the computation instance.


Builder method
^^^^^^^^^^^^^^

TODO



Distribution and deployment
----------------------------

Your module must also provide a a few predefined utility methods, used to
build and deploy your algorithm, and may define some optional methods, used for
deployment and distribution.

Most distribution methods require the following two methods. These methods
are generally required for a correct distribution of the computations on
agents, but if you only want to use `oneagent` distribution (or simply
during development) you can simply return 0::

     def computation_memory(computation: ComputationNode, links):
     """
       This method must return the memory footprint for the given computation
       from the graph.
     """

::

    def communication_load(link: Link):
    """
    This method must return the communication load for this link in the
    computation graph.
    """


When deploying  the computation, concrete ``MessagePassingComputation`` objects
must be instantiated on their assigned agent. For this, an algorithm
module **must** also provide a factory method to build computation object::

    def build_computation(node: ComputationNode, links: Iterable[Link], algo: AlgorithmDef)-> MessagePassingComputation:
    """
    Build a computation instance for a given algorithm (and parameters)
    """



Computations's footprint
^^^^^^^^^^^^^^^^^^^^^^^^

TODO

Communication load
^^^^^^^^^^^^^^^^^^

TODO
