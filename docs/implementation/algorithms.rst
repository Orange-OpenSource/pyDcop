

.. _implementation_algorithms:

Implementing a DCOP algorithm
=============================

By providing all the infrastructure, pyDCOP makes it easier to implement a
new DCOP algorithm ; you only have one python module to implement,
with only one mandatory class.


To implement an algorithm you need:

- to create a python module
- to define one or several :py:class:`.Message`
- to implement your logic in a
  :py:class:`.DcopComputation` class
- to implement a few predefined utility methods and attribute
- optionally, you may also implement some other methods used for some
  distribution methods


Module
------

An algorithm must be defined in it's own module in :py:mod:`pydcop.algorithms`.
For example, `dsa` is implemented in the module :py:mod:`pydcop.algorithm.dsa`.
The name of the module is the name you will use
when running your algorithm with the ``pydcop``
command line interface (``-a`` or ``--algo`` parameter).
For example, for a new algorithm named ``my_algorithm``,
you simply create a file ``my_algorithm.py`` in ``pydcop/algorithms``.
You will then be able to use this algorithm
to solve a DCOP with the following command::

  pydcop solve --algo my_algorithm [...]


Messages
--------

DCOP algorithms are *message passing* algorithms: they work by sending
messages to each other. You must define the messages used by your algorithm.
The easiest approach is to use the
:py:mod:`pydcop.infrastructure.computations.message_type`
class factory method to define your message(s).

For example, the following will define a message type ``MyMessage``, with two
fields ``foo`` and ``bar``::

  MyMessage = message_type('MyMessage', ['foo', 'bar'])

You can then use ``MyMessage`` like any class::

  >>> msg1 = MyMessage(foo=42, bar=21)
  >>> msg.foo
  42
  >>> msg.type
  'MyMessage'

You can also subclass :py:mod:`pydcop.infrastructure.computations.Message`,
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

In any case, you **must** use the
:py:mod:`pydcop.utils.simple_repr.SimpleRepr` mixin
(:py:mod:`pydcop.infrastructure.computations.Message` already extends it)
for your message to be serializable.
This is necessary when running the agents in
different processes, as messages will be sent over the network.


Computation
-----------

An algorithms consists in one or several :py:class:`DcopComputation` class.
Most algorithms have one single type of computation, which is
responsible for selecting the value for a single variable.
In this case you should subclass :py:class:`VariableComputation`,
which provides some convenient methods for value selection.

For more complex algorithm, you can define several computations
(with pyDCOP, your algorithm can have as many kind of computation as you want),
look at MaxSum's implementation for an example
(`MaxSum` has two kind of computations, for `Factor` and `Variable`)..


Receiving messages
^^^^^^^^^^^^^^^^^^

At runtime, an instance of a computation is deployed on an agent,
which notify it when receiving a message.
The computation then process the message and,
if necessary, emits new messages for other computations.

For each message type, you must declare a handler method::

  def __init__(self, variable, comp_def)
      super().__init__(variable, comp_def)
      self._msg_handlers['my_message'] = self._on_my_message

  ...

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
Messages are sent by calling ``self.post_msg`` ::

  self.post_msg(target_computation_name, message_object)

All computations must be subclasses of ``MessagePassingComputation``.
In each of these classes implements the ``on_message`` method to handle
received message. Alternatively, you may also extend the
``AbstractMessagePassingAlgorithm`` class and register one method for
each of the message in your constructor::

    super().__init__()
    self._msg_handlers['msg_type'] = self._on_my_msg

Selecting a value
^^^^^^^^^^^^^^^^^

In your computation, when selecting a value for a variable, you **must**
call ``self.value_selection`` with the value and the associated local cost.
This is allows pydcop to monitor value selection on each agent and
extract the final assignment::

    self.value_selection(self._v.initial_value, None)


Cycles
^^^^^^



Various
-------

* finishing a computation : using a `finished` signal

* changing cycle

Parameters
^^^^^^^^^^

If the algorithm supports parameters, you must define a method to parse the
command-line supplied parameters and set default values for parameters that
were not given::

    def algo_params(params: Dict[str, str]):
    """
    Returns the parameters for the algorithm.

    If a value for parameter is given in `params` it is parsed and (if acceptable)
    used, otherwise a default value is used instead.

    :param params: a dict containing name and values for parameters as string
    :return:a Dict with all parameters (either their default value or
    the values extracted form `params`)
    """

An ``Algodef`` instance populated with the parsed parameter will be passed to
your ``build_computation`` method, you can then use it to pass these parameter
to the computation instance.


Distribution and deployement
----------------------------


Your module must also provide a a few predefined utility methods, used to
build and deploy your algorithm, and may define some optional method, used for
deployement and distribution.

The module of your algorithm **must** also an attribute named ``GRAPH_TYPE`` which
must contains the name of the computation graph type used. Available
computation graph types are ``'factor_graph'``, ``'pseudo_tree'`` and
``'constraints_hypergraph'``, other could be defined in the future::

    GRAPH_TYPE = 'constraints_hypergraph'

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
must be instanciated on their assignated agent. For this, an algorithm
module **must** also provide a factory method to build computation object::

    def build_computation(node: ComputationNode, links: Iterable[Link], algo: AlgoDef)-> MessagePassingComputation:
    """
    Build a computation instance for a given algorithm (and parameters)
    """



