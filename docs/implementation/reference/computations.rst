
.. _implementation_reference_computations:

pydcop.infrastructure.computations
==================================

.. automodule:: pydcop.infrastructure.computations

.. autoclass:: pydcop.infrastructure.computations.Message
  :members:

.. autofunction:: pydcop.infrastructure.computations.message_type

.. autofunction:: pydcop.infrastructure.computations.register

.. autoclass:: pydcop.infrastructure.computations.MessagePassingComputation
  :members: on_start, on_pause, on_stop, post_msg, add_periodic_action, remove_periodic_action

.. autoclass:: pydcop.infrastructure.computations.DcopComputation
  :members: neighbors, new_cycle, cycle_count, footprint, post_to_all_neighbors

.. autoclass:: pydcop.infrastructure.computations.VariableComputation
  :members: variable, current_value, current_cost, value_selection, random_value_selection