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
DBA : Distributed Breakout Algorithm
------------------------------------

The Distributed Breakout algorithm :cite:`yokoo_distributed_1996` is a local-search
algorithm,
built as a a distributed version of the Breakout algorithm for CSP
:cite:`morris_breakout_1993`.
It is meant to solve distributed constraints satisfaction
problems (and not optimization problems).

See also :cite:`wittenburg_distributed_2003` on using DBA for optimization
problems.

Algorithm Parameters
^^^^^^^^^^^^^^^^^^^^

Our DBA implementation supports two parameters:

* **infinity**: the value used as 'infinity', returned as the cost of a
  violated constraint (it must map the value used in your dcop definition).
  Defaults to 10 000

* **max_distance** : an upper bound for the maximum distance between two agents
  in the graph. Ideally you would use the graph diameter or simply the number
  of variables in the problem. It is used for termination detection (which in
  DBA only works is there is a solution to the problem). Defaults to 50

Example
^^^^^^^
::

    pydcop -t 2 solve -a dba -p infinity:10000 max_distance:3 \
           -d adhoc graph_coloring_csp.yaml
    {
      "assignment": {
        "v1": "G",
        "v2": "R",
        "v3": "G"
      },
      "cost": 0,
      "duration": 1.9932785034179688,
      "status": "TIMEOUT"
    }


Messages
^^^^^^^^

.. autoclass:: DbaOkMessage
  :members:

.. autoclass:: DbaImproveMessage
  :members:

.. autoclass:: DbaEndMessage
  :members:

Computation
^^^^^^^^^^^

.. autoclass:: DbaComputation
  :members:



"""
import logging
import random

from typing import Iterable, Dict

from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.infrastructure.computations import Message, VariableComputation, \
    register

from pydcop.computations_graph.constraints_hypergraph import \
    VariableComputationNode
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import RelationProtocol, filter_assignment_dict

INFINITY = 10000

HEADER_SIZE = 100
UNIT_SIZE = 5


# Type of computations graph that must be used with dsa
GRAPH_TYPE = 'constraints_hypergraph'

def build_computation(comp_def: ComputationDef) -> VariableComputation:
    return DbaComputation(comp_def.node.variable,
                          comp_def.node.constraints,
                          mode=comp_def.algo.mode,
                          **comp_def.algo.params,
                          comp_def=comp_def)


def computation_memory(computation: VariableComputationNode) -> float:
    """Return the memory footprint of a DBA computation.

    Notes
    -----
    With DBA, a computation must only remember the current value for each
    of it's neighbors.

    Parameters
    ----------
    computation: VariableComputationNode
        a computation in the hyper-graph computation graph

    Returns
    -------
    float:
        the memory footprint of the computation.

    """
    neighbors = set((n for l in computation.links for n in l.nodes
                     if n not in computation.name))
    return len(neighbors) * UNIT_SIZE


def communication_load(src: VariableComputationNode, target: str) -> float:
    """Return the communication load between two variables.

    Notes
    -----
    The main messages in DBA are the 'ok?' and 'improve' messages, which at
    most contains a value and a possible improvement. The size of the message
    does not depends on the source nor target variable, nor on their
    respective domains.

    Parameters
    ----------
    src: VariableComputationNode
        The ComputationNode for the source variable.
    target: str
        the name of the other variable `src` is sending messages to

    Returns
    -------
    float
        The size of messages sent from the src variable to the target variable.
    """
    return 2 * UNIT_SIZE + HEADER_SIZE


# ###########################   MESSAGES   ################################
class DbaOkMessage(Message):
    def __init__(self, value):
        super().__init__('dba_ok', None)
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return 1

    def __str__(self):
        return 'DbaOkMessage({})'.format(self.value)

    def __repr__(self):
        return 'DbaOkMessage({})'.format(self.value)

    def __eq__(self, other):
        if type(other) != DbaOkMessage:
            return False
        if self.value == other.value:
            return True
        return False


class DbaImproveMessage(Message):
    def __init__(self, improve, current_eval, termination_counter):
        super().__init__('dba_improve', None)
        self._improve = improve
        self._current_eval = current_eval
        self._termination_counter = termination_counter

    @property
    def current_eval(self):
        return self._current_eval

    @property
    def improve(self):
        return self._improve

    @property
    def termination_counter(self):
        return self._termination_counter

    @property
    def size(self):
        return 1

    def __str__(self):
        return 'DbaImproveMessage(improve:{}, eval: {})'.format(
            self.improve, self.current_eval)

    def __repr__(self):
        return 'DbaImproveMessage({}, {})'.format(self.improve,
                                                  self.current_eval)

    def __eq__(self, other):
        if type(other) != DbaImproveMessage:
            return False
        if (self.improve == other.improve) and \
                (self.current_eval == other.current_eval):
            return True
        return False


class DbaEndMessage(Message):
    def __init__(self):
        super().__init__('dba_end', None)

    @property
    def size(self):
        return 1

    def __str__(self):
        return 'DbaEndMessage()'

    def __repr__(self):
        return 'DbaEndMessage()'

    def __eq__(self, other):
        return type(other) == DbaEndMessage


algo_params = [
    AlgoParameterDef('infinity', 'int', None, 10000),
    AlgoParameterDef('max_distance', 'int', None, 50),
]


# ###########################   COMPUTATION   ############################
class DbaComputation(VariableComputation):
    """
    DBAComputation implements DBA.

    See. the following article for a description of original DBA: 'Distributed
    Breakout Algorithm for Solving Distributed Constraint Satisfaction
    Problems' (Makoto Yokoo, Katsutoshi Hirayama, 1996)
    """

    def __init__(self, variable: Variable,
                 constraints: Iterable[RelationProtocol],
                 msg_sender=None, mode='min',
                 infinity=INFINITY, max_distance=50,
                 comp_def=None):
        """
        :param variable: a variable object for which this computation is
        responsible
        :param constraints: the list of constraints involving this variable
        :param max_distance: The distance to the furthest agent in the
        constraint graph, or an appropriate upper bound. Defaults to 50
        """

        super().__init__(variable, comp_def)
        if mode != 'min':
            raise ValueError('DBA is a constraint **satisfaction** '
                             'algorithm and only support '
                             'minimization objective')

        self._msg_sender = msg_sender

        global INFINITY
        INFINITY = infinity
        self._max_distance = max_distance

        # Handling messages arriving during wrong mode
        self.__postponed_improve_messages__ = []
        self.__postponed_ok_messages__ = []

        self.__constraints__ = list(constraints)
        self.__constraints_weights__ = [1 for _ in constraints]
        self._violated_constraints = []
        # The algorithm starts in "ok?" mode
        self._mode = 'starting'
        # some constraints might be unary, and our variable can have several
        # constraints involving the same variable
        self._neighbors = set([v.name for c in constraints
                               for v in c.dimensions if v != variable])
        # Agent view of its neighbors resp. for ok and improve modes
        self._neighbors_values = {}
        self._neighbors_improvements = {}
        # The following values are calculated further by the algorithm as
        # described in the first article mentionned above. However, we set
        # them as class attribute so that they can be accessible from both
        # modes (ok and improve)
        self._termination_counter = 0
        self._consistent = None  # no constraints violated ?
        self._can_move = False  # agent is allowed to change its value
        self._quasi_local_minimum = False
        self._my_improve = 0  # Possible improvement the agent can realize
        self._new_value = None

    @property
    def constraints(self):
        return self.__constraints__

    @property
    def neighbors(self):
        return self._neighbors

    def on_start(self):
        # randomly select a value
        self.value_selection(random.choice(self.variable.domain),
                             self.current_cost)
        self.logger.info('%s dba starts: randomly select value %s and '
                         'send to neighbors', self.variable.name,
                         self.current_value)
        self._send_current_value()
        self._go_to_wait_ok_mode()

    @register("dba_ok")
    def _on_ok_msg(self, variable_name, recv_msg, _):
        """
        This method  implements the wait_ok_mode of DBA as described in
        'Distributed Breakout Algorithm for Solving Distributed Constraint
        Satisfaction Problems' (Makoto Yokoo, Katsutoshi Hirayama, 1996)
        """
        if self._mode == 'ok':
            self._handle_ok_message(variable_name, recv_msg)
        else:
            # A ok? message can be received during the improve mode (due to
            # async.). In this case, its handling s postponed until the next
            # iteration of wait_ok_mode
            self.__postponed_ok_messages__.append((variable_name, recv_msg))

    def _handle_ok_message(self, variable_name, recv_msg):

        self._neighbors_values[variable_name] = recv_msg.value
        self.logger.info('%s received variable value %s from %s',
                          self.variable.name, recv_msg, variable_name)
        # if we have a value for all neighbors, compute our best value for
        # conflict reduction
        if len(self._neighbors_values) == len(self._neighbors):
            self.logger.info('%s received OK values from all neighbors : %s',
                              self.name,
                              self._neighbors_values)
            # Replace all variables except its own variable with the values
            # received from neighbors
            reduced_cs = []
            for c in self.constraints:
                asgt = filter_assignment_dict(self._neighbors_values,
                                              c.dimensions)
                reduced_cs.append(c.slice(asgt))

            self.__cost__, _ = self.compute_eval_value(self.current_value,
                                                          reduced_cs)
            # Compute and send best improvement to neighbors
            self.improve(reduced_cs)

            self._go_to_wait_improve_mode()
        else:
            # Still waiting for other neighbors
            self.logger.info(
                '%s waiting for OK values from other neighbors (got %s) but '
                'neighbors are %s',
                self.name, self._neighbors_values, self.neighbors)

    def improve(self, relations):
        current_eval = self.__cost__
        bests, best_eval = self._compute_best_improvement(relations)

        if current_eval == 0:
            self._consistent = True
        else:
            self._consistent = False
            self._termination_counter = 0

        self._my_improve = current_eval - best_eval
        if self._my_improve > 0:
            self._can_move = True
            self._quasi_local_minimum = False
            self._new_value = random.choice(bests)
        else:
            self._can_move = False
            self._quasi_local_minimum = True

        _, self._violated_constraints = self.compute_eval_value(
            self.current_value, relations)

        self._send_improve(current_eval)

    def _send_improve(self, current_eval):
        msg = DbaImproveMessage(self._my_improve, current_eval,
                                self._termination_counter)
        for n in self.neighbors:
            self.post_msg(n, msg)

    def _compute_best_improvement(self, relations):
        """
        :param: the reduced constraints for the variable, so that only its value
         is not set

        :return: (list of values achieving best improvement, best improvement)
        """
        best_vals = []
        best_eval = INFINITY
        for v in self.variable.domain:
            curr_eval, _ = self.compute_eval_value(v, relations)
            if curr_eval < best_eval:
                best_eval = curr_eval
                best_vals = [v]
            elif curr_eval == best_eval:
                best_vals.append(v)

        return best_vals, best_eval

    def _send_current_value(self):
        for n in self._neighbors:
            msg = DbaOkMessage(self.current_value)
            self.post_msg(n, msg)

    def compute_eval_value(self, val, relations):
        """
        This function compute the evaluation value (the number of violated
        constraints) regarding the current assignment.

        Parameters
        ----------
        val: Any
            A value for the variable of this object. You can choose any
            of the definition domain, according to the context in which you use
            the function.

        relations: list of constraints objects
            The list of constraints involving the variable of this
            computation, with the values of other variables set to the values
            sent by the neighbors

        Returns
        -------
        The evaluation value for the given assignment and the list
        of indices of the violated constraints for this value
        """
        i = 0
        new_eval_value = 0
        violated_constraints = []
        for rel in relations:
            if rel(val) >= INFINITY:
                violated_constraints.append(i)
                new_eval_value += self.__constraints_weights__[i]
            i += 1
        return new_eval_value, violated_constraints

    def _go_to_wait_improve_mode(self):
        self._mode = 'improve'
        # if improve messages were received during wiat_ok_mode, they should be
        # processed now
        self.logger.info('%s entering improve mode', self.name)
        for sender, msg in self.__postponed_improve_messages__:
            self._handle_improve_message(sender, msg)
        self.__postponed_improve_messages__.clear()


# #############################IMPORVE MODE##################################
    @register("dba_improve")
    def _on_improve_msg(self, variable_name, recv_msg, _):

        if self._mode == 'improve':
            self._handle_improve_message(variable_name, recv_msg)
        else:
            self.__postponed_improve_messages__.append(
                (variable_name, recv_msg))

    def _handle_improve_message(self, variable_name, recv_msg):
        self._neighbors_improvements[variable_name] = recv_msg
        self.logger.info('%s received possible improvement value %s from %s',
                         self.name, recv_msg.improve, variable_name)

        self._termination_counter = min(recv_msg.termination_counter,
                                        self._termination_counter)

        if recv_msg.improve > self._my_improve:
            self._can_move = False
            self._quasi_local_minimum = False
        elif recv_msg.improve == self._my_improve and self.name > variable_name:
            self._can_move = False

        if recv_msg.current_eval > 0:
            self._consistent = False
        # if messages received from all neighbors
        if len(self._neighbors_improvements) == len(self._neighbors):
            # determine if can change value and send ok message to neighbors
            self._send_ok()
            # End of a cycle: clear agent view
            self._neighbors_improvements.clear()
            self._neighbors_values.clear()
            self._violated_constraints.clear()

            self._go_to_wait_ok_mode()
        else:
            # Still waiting for other neighbors
            self.logger.info(
                '%s waiting for IMPROVE values from other neighbors (got %s) '
                'and neighbors are %s',
                self.name, self._neighbors_improvements, self.neighbors)

    def _send_ok(self):
        self.new_cycle()
        stop = False
        if self._consistent:
            self._termination_counter += 1
            stop = self.stop_condition()

        if stop:
            self._send_end_msg()
            self._mode = 'finished'
            self.logger.debug('%s has finished its computation for DBA',
                              self.name)
            self.finished()
        else:
            if self._quasi_local_minimum:
                self._increase_weights(self._violated_constraints)
                # To adapt DBA to the current code context, we choose to use the
                # eval value as the "cost" used for dcop algorithms since this
                # is the cost used to evaluate if the variable vlue can change
            if self._can_move:
                self.value_selection(self._new_value,
                                     self.__cost__ - self._my_improve)

            msg = DbaOkMessage(self.current_value)
            for n in self._neighbors:
                self.post_msg(n, msg)

    def _increase_weights(self, constraints):
        self.logger.info('Increasing the weights of the constraints %s',
                            constraints)
        for i in constraints:
            self.__constraints_weights__[i] += 1

    def _go_to_wait_ok_mode(self):
        self._mode = 'ok'
        self.logger.info('%s entering ok mode', self.name)
        for sender, msg in self.__postponed_ok_messages__:
            self._handle_ok_message(sender, msg)
        self.__postponed_ok_messages__.clear()

    @register("dba_end")
    def _on_end_msg(self, variable_name, recv_msg, _):
        # To avoid to send again and again EndMessages to the neighbors
        if self._mode != 'finished':
            self._send_end_msg()
            self._mode = 'finished'
            self.finished()

    def _send_end_msg(self):
        msg = DbaEndMessage()
        for n in self.neighbors:
            self.post_msg(n, msg)

    def stop_condition(self):
        return self._termination_counter == self._max_distance

    def _in_wait_ok_mode(self):
        return self._mode == 'ok'

    def _in_wait_improve_mode(self):
        return self._mode == 'improve'
