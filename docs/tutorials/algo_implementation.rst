
.. _tutorials_algorithm_implementation:


Implementing a DCOP algorithm
=============================

One of pyDCOP goals is to help the study and development of DCOP algorithms.
For this purpose, pyDCOp allows you to implement easily your own algorithms
and provide all the required infrastructure so that you can concentrate
on your algorithm logic.

To demonstrate this we will implement a very simple DCOP algorithm in this
tutorial, the Distributed Stochastic Algorithm (DSA).

Distributed Stochastic Algorithm
--------------------------------

DSA is a synchronous stochastic local search algorithm that works on a
constraints graph.
At startup, each variable takes a random value from it's domain
and then run the same procedure in repeated steps.

At each step, each variable send its value to its neighbors.
Once a variable has received the value from all it's neighbors,
it evaluates the gain it could obtain by picking another value.
If this gain is positive, it decides to change it's value
or to keep the current one.
This decision is made stochastically: a variable change its value with
probability **p** (if doing so can improve the state quality).

The algorithm stops after a predefined number of steps.

Note
^^^^

For example purpose, we only consider here a a very simple version of DSA,
which implements DSA-A as described in :cite:`zhang_distributed_2005`.
pyDCOP also has a full implementation of DSA, with several variants,
both synchronous and asynchronous, which can be used as ``dsa`` and ``adsa``
in the :ref:`solve<pydcop_commands_solve>`, :ref:`run<pydcop_commands_run>` and
:ref:`orchestrator<pydcop_commands_orchestrator>` commands.


Implementation with pyDCOP
--------------------------

Basic setup
^^^^^^^^^^^

In pyDCOP, each algorithm is implemented as a module
in the ``pydcop.algorithms`` package.
Thus for our DSA implementation, we simply create a ``dsa_tuto.py`` file
in the directory ``pydcop/algorithms``.

We must then declare wich :ref:`graph model<concepts_graph>`
our algorithm works on. This is done by declaring a ``GRAPH_TYPE`` variable::

  # Type of computations graph that must be used with dsa-tuto
  GRAPH_TYPE = 'constraints_hypergraph'

Then we need to define the message(s) for this algorithm.
A message is defined by a class that inherits
:class:`pydcop.infrastructure.computations.Message`
but you can use the :func:`.message_type` function,
which creates the subclass automatically for you.
DSA uses one single type of message, which contains the value of the variable
sending the message::

  DsaMessage = message_type("dsa_value", ["value"])


Algorithm implementation class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generally speaking, an algorithm is implemented as a subclass of
:class:`.DcopComputation`.
For algorithms that define computations for variables
(i.e. most classic algorithms),
you must subclass :class:`.VariableComputation`, which defines
several utility methods for handling variables::

  class DsaTutoComputation(VariableComputation):

One instance of ``DsaTutoComputation`` will be created by pyDCOP
for each variable in our DCOP.

The constructor of our class must accept
a :class:`.ComputationDef` instance as argument.
As its name implies, a :class:`.ComputationDef` instance is an object
that fully describes a computation an can be used to instanciate one.
It is made of a :class:`.ComputationNode` and a :class:`.AlgorithmDef`.
You don't need to bother with this for now, these instances will
be automatically be created by pyDCOP and passed to your constructor.
With this, we can now write our computation's constructor::

  class DsaTutoComputation(VariableComputation):

      def __init__(self, computation_definition: ComputationDef):
          # Always call the super class constructor !
          super().__init__(computation_definition.node.variable,
                           computation_definition)

          # Constraints involving this variable are available on the
          # ComputationNode:
          self.constraints = computation_definition.node.constraints

          # The assignment of our neighbors for the current and next cycle
          self.current_cycle = {}
          self.next_cycle = {}

Startup
^^^^^^^

When pyDCOP starts a computation its ``on_start`` method is automatically called.
You can use it for any startup logic.
In the case of DSA, the computation must pick a value for the variable
it represents::

    def on_start(self):
        # This picks a random value form the domain of the variable
        self.random_value_selection()

        # The currently selected value is available through self.current_value.
        self.post_to_all_neighbors(DsaMessage(self.current_value))
        self.evaluate_cycle()  # Defined later


Message handling
^^^^^^^^^^^^^^^^

Once started, computations communicate one with another by sending messages.
In order to handle the messages sent to by the computation, you must
register a message handler using the :func:`.register` decorator :
``@register("dsa_value")``.

For DSA, when receiving a message, we store the value and check
if we received a value from all our neighbors,
in which case we can evaluate
whether we should pick a new value for our variable.
Note that there might be an offset of one cycle with our neighbor.

Here is the corresponding message handler::

    @register("dsa_value")
    def on_value_msg(self, variable_name, recv_msg, t):

        if variable_name not in self.current_cycle:
            self.current_cycle[variable_name] = recv_msg.value
            if self.is_cycle_complete():
                self.evaluate_cycle()

        else:  # The message for the next cycle
            self.next_cycle[variable_name] = recv_msg.value


Finally, we can decide if we should select another value
by computing the potential gain and drawing a random number::

    def evaluate_cycle(self):

        self.current_cycle[self.variable.name] = self.current_value
        current_cost = assignment_cost(self.current_cycle, self.constraints)
        arg_min, min_cost = self.compute_best_value()

        if current_cost - min_cost > 0 and 0.5 > random.random():
            # Select a new value
            self.value_selection(arg_min)

        self.current_cycle, self.next_cycle = self.next_cycle, {}
        self.post_to_all_neighbors(DsaMessage(self.current_value))

    def is_cycle_complete(self):
        # The cycle is complete if we received a value from all the neighbors:
        return len(self.current_cycle) == len(self.neighbors)

    def compute_best_value(self) -> Tuple[Any, float]:
        # compute the best possible value and associated cost
        arg_min, min_cost = None, float('inf')
        for value in self.variable.domain:
            self.current_cycle[self.variable.name] = value
            cost = assignment_cost(self.current_cycle, self.constraints)
            if cost < min_cost:
                min_cost, arg_min = cost, value
        return arg_min, min_cost


Running the algorithm
^^^^^^^^^^^^^^^^^^^^^

You now have a full working implementation of DSA.
For reference, this implementation is also available in this file :
:download:`dsa-tuto.py<dsa-tuto.py>`.

If you did not follow the tutorial, you can simply copy this file in
``pydcop/algorithms``.

This implementation can be used with any pydcop command,
for example for solving a graph coloring DCOP for 50 variables
(:download:`graph_coloring_50.yaml<graph_coloring_50.yaml>`)
you can use::

  pydcop --timeout 10 -v 3 solve --algo dsa-tuto graph_coloring_50.yaml

Note that this tutorial only covers
the basics of DCOP algorithms implementation,
for more details, look at :ref:`implementation_algorithms`.
