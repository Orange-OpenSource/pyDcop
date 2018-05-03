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


import logging
import random
import operator as op

import functools as fp
from collections import defaultdict
from typing import Iterable, Dict, Any, Tuple, List

from pydcop.algorithms import filter_assignment_dict, \
    generate_assignment_as_dict, ComputationDef
from pydcop.infrastructure.computations import Message, VariableComputation

from pydcop.computations_graph.constraints_hypergraph import \
    VariableComputationNode
from pydcop.dcop.objects import Variable
from pydcop.dcop.relations import RelationProtocol, find_dependent_relations

__author__ = "Pierre Nagellen, Pierre Rust"

GRAPH_TYPE = 'constraints_hypergraph'

HEADER_SIZE = 100
UNIT_SIZE = 5


def algo_name() -> str:
    """

    Returns
    -------
    The name of the algorithm implemented by this module : 'mgm2'
    """
    return __name__.split('.')[-1]


def build_computation(comp_def: ComputationDef) -> VariableComputation:
    return Mgm2Computation(comp_def.node.variable,
                           comp_def.node.constraints,
                           mode=comp_def.algo.mode,
                           **comp_def.algo.params,
                           comp_def=comp_def)


def computation_memory(computation: VariableComputationNode) -> float:
    """Return the memory footprint of a MGM2 computation.

    Notes
    -----
    With MGM2, a computation must remember the current value and gain for each
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
    return len(neighbors) * 2 * UNIT_SIZE


def communication_load(src: VariableComputationNode, target: str) -> float:
    """Return the communication load between two variables.

    Notes
    -----
    The biggest messages in MGM2 is the 'offer' message, which contains a
    map of coordinated moves and associated gains.

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
    target_v = None
    for c in src.constraints:
        for v in c.dimensions:
            if v.name == target:
                target_v = v
    if not target_v:
        raise ValueError('target variable {} not found in constraints for {}'
                         .format(target, src))

    nb_pairs = len(target_v.domain) * len(src.variable.domain)

    # for potential coordinated move we have two value and a gain :
    return nb_pairs * UNIT_SIZE * 3 + HEADER_SIZE


def algo_params(params: Dict[str, str]):
    """
    Returns the parameters for the algorithm.

    If a value for parameter is given in `params` it is used, otherwise a
    default value is used instead.

    :param params: a dict containing name and values for parameters
    :return:
    """
    mgm2_params = {
        'threshold': 0.5,
        'favor': 'unilateral',
        'cycle_stop': None
    }
    if 'threshold' in params:
        try:
            mgm2_params['threshold'] = float(params['threshold'])
        except ValueError:
            raise ValueError("'threshold' parameter for MGM2 must be a float")
    if 'favor' in params:
        if params['favor'] in ['unilateral', 'no', 'coordinated']:
            mgm2_params['favor'] = params['favor']
        else:
            raise ValueError("'favor' parameter for MGM2 must be "
                             "'unilateral', 'no' or 'coordinated'")
    if 'cycle_stop' in params:
        try:
            mgm2_params['cycle_stop'] = int('cycle_stop')
        except ValueError:
            raise ValueError("''cycle_stop' parameter must be an int")

    remaining_params = set(params) - {'threshold', 'favor', 'cycle_stop'}
    if remaining_params:
        raise ValueError('Unknown parameter(s) for MGM2 : {}'
                         .format(remaining_params))
    return mgm2_params


# ############################   MESSAGES   ################################
class Mgm2ValueMessage(Message):
    """
    Class to send a message informing neighbors of the agent value

    """

    def __init__(self, value):
        super().__init__('value', None)
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return 1

    def __str__(self):
        return 'Mgm2ValueMessage({})'.format(self.value)

    def __repr__(self):
        return 'Mgm2ValueMessage({})'.format(self.value)

    def __eq__(self, other):
        if type(other) != Mgm2ValueMessage:
            return False
        if self.value == other.value:
            return True
        return False


# Basically the same class than Mgm2ValueMessage, but we need two classes to
# differentiate the kind of messages received for postponing processing when
# not in the good state
class Mgm2GainMessage(Message):
    """
    Class to send to neighbors what best gain can be achieved by the agent if
    it moves alone

    """

    def __init__(self, value):
        """
        :param value: max gain of the agent
        """
        super().__init__('gain', None)
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return 1

    def __str__(self):
        return 'Mgm2GainMessage({})'.format(self.value)

    def __repr__(self):
        return 'Mgm2GainMessage({})'.format(self.value)

    def __eq__(self, other):
        if type(other) != Mgm2GainMessage:
            return False
        if self.value == other.value:
            return True
        return False


class Mgm2OfferMessage(Message):
    """
    Class to send an offer message to a neighbor.
    The value of the message is a dictionary which keys are couples
    representing coordinate moves (sender_val, receiver_val), and which
    values are the gain realized by the offerer.
    E.g. {(1,0): 5, (0,1): 2}

    Moreover, to handle asynchronicity, we add the following fact. Each agent
    send an offer message to all its neighbors. The agent distinguish a real
    offer thanks to the 'offer' attribute. A real (fake) offer has 'offer' set
    to '(not) offering'. This artifice allows every agents to know if they have
    received all the offers they should before processing next step.
    """

    def __init__(self, offers: Dict[Tuple[Any, Any], float]=None,
                 is_offering=False):
        super().__init__('offer', None)
        self._offers = offers if offers is not None else dict()
        self._is_offering = is_offering

    @property
    def offers(self) -> Dict[Tuple[Any, Any], float]:
        if self._offers is None:
            return dict()
        return self._offers

    @property
    def is_offering(self):
        return self._is_offering

    @property
    def size(self):
        return 3 * len(self._offers)

    def _simple_repr(self):
        r = {'__module__': self.__module__,
             '__qualname__': self.__class__.__qualname__,
             'is_offering': self.is_offering,
             'var_values': list(),
             'gains': list()}

        # When building the simple repr we transform the dict into a pair
        # of list to avoid problem when serializing / deserializing the repr.
        # The costs dic often contains int as key, when converting to an from
        # json (which only support string for keys in dict), we would
        # otherwise loose the type information and restore the dict with str
        # keys.

        # var_values : [ (value_var_sender, value_var_receiver), ...]
        # gains :[ gain_first_pair, ...]

        if self.is_offering:
            if self.offers:
                var_values, gains = zip(*self.offers.items())
                r['var_values'] = var_values
                r['gains'] = gains

        return r

    @classmethod
    def _from_repr(cls, r):
        if 'gains' in r:
            var_values = [tuple(couple) for couple in r['var_values']]
            gains = r['gains']

            return Mgm2OfferMessage(dict(zip(var_values, gains)),
                                    r['is_offering'])

        return Mgm2OfferMessage(dict(), r['is_offering'])

    def __str__(self):
        return 'Mgm2OfferMessage({},{})'.format(self.is_offering, self.offers)

    def __repr__(self):
        return 'Mgm2OfferMessage({},{})'.format(self.is_offering, self.offers)

    def __eq__(self, other):
        if type(other) != Mgm2OfferMessage:
            return False
        if self.offers == other.offers:
            return True
        return False


class Mgm2ResponseMessage(Message):
    """
    Class to send a response message to an offer made by a neighbor

    """

    def __init__(self, accept: bool, value=None, gain=None):
        """
        :param accept: True for 'accept', False for 'reject'
        :param value: the value of the neighbor for the accepted offer (if so)
        :param gain: the global gain realized thanks to the accepted offer (
        if so)
        """
        super().__init__('answer?', None)

        self._accept = accept
        if accept:
            if (value is None) or (gain is None):
                raise ValueError("If you send an accept message, you must send"
                                 "the neighbor value and the global gain in "
                                 "it too")
            self._value = value
            self._gain = gain
        else:
            self._value = None
            self._gain = None

    @property
    def accept(self):
        return self._accept

    @property
    def value(self):
        return self._value

    @property
    def gain(self):
        return self._gain

    @property
    def size(self):
        return 3

    def __str__(self):
        return 'Mgm2ResponseMessage({},{})'.format(self.accept, self.value,
                                                   self._gain)

    def __repr__(self):
        return 'Mgm2ResponseMessage({},{})'.format(self.accept,
                                                   self.value, self._gain)

    def __eq__(self, other):
        if type(other) != Mgm2ResponseMessage:
            return False
        if self.accept == other.accept and \
                self.value == other.value and self.gain == other.gain:
            return True
        return False


class Mgm2GoMessage(Message):
    """
    Class to send my commited partner if we can change our values or not

    """

    def __init__(self, go: bool):
        super().__init__('go?', None)
        self._go = go

    @property
    def go(self):
        return self._go

    @property
    def size(self):
        return 1

    def __str__(self):
        return 'Mgm2GoMessage({})'.format(self.go)

    def __repr__(self):
        return 'Mgm2GoMessage({})'.format(self.go)

    def __eq__(self, other):
        if type(other) != Mgm2GoMessage:
            return False
        if self.go == other.go:
            return True
        return False


# ###########################   COMPUTATION   ############################
class Mgm2Computation(VariableComputation):
    """
    Mgm2Computation implements Mgm2 algorithm as described in 'Distributed
    Algorithms for DCOP: A Graphical-Game-Base Approach' (R. Maheswaran,
    J. Pearce, M. Tambe, 2004)

    Warning: The attribute _neighbors is the list of the variable neighbors as
    Variable objects, while the property neighbors gives access to the list of
    the names of neighbors' variables.

    Parameters
    ----------

    variable: Variable object
        a variable object for which this computation is responsible.
    constraints: Iterable[RelationProtocol]
        the list of utilities/constraints involving this variable.
    threshold: float
        the threshold under which the agent is an offerer. This must be
        between 0 and 1.
    mode: str
        optimization mode 'min' or 'max'. Defaults to 'min'
    msg_sender: a message sender
        used to send messages
    logger: a logger
        used to log messages
    favor: 'unilateral',
        the type of moved that is favored in the algorithm : 'unilateral', 'no'
        or 'coordinated'
    cycle_stop: int
        number of cycles before stopping. If None, the computation does not
        stop autonomously.
    comp_def: ComputationDef
        The computation definition this computation has been built from.

    """

    def __init__(self, variable: Variable,
                 constraints: Iterable[RelationProtocol],
                 threshold: float =0.5,
                 mode: str='min',
                 msg_sender=None,
                 favor: str='unilateral', cycle_stop: int=None,
                 comp_def: ComputationDef=None):

        super().__init__(variable, comp_def)
        # MGM2 a 5 different states, each with a specific handler method:
        self.states = {
            'value': self._handle_value_message,
            'offer': self._handle_offer_msg,
            'answer?': self._handle_response_msg,
            'gain': self._handle_gain_message,
            'go?': self._handle_go_message
        }

        self._msg_sender = msg_sender
        self.logger = logging.getLogger('pydcop.algo.mgm2.' + variable.name)
        self.cycle_stop = cycle_stop

        # Handling messages arriving during wrong mode
        self._postponed_msg = defaultdict(lambda: [])  # type: Dict[str, List]

        self._partner = None
        self._committed = False
        self._is_offerer = False
        self._threshold = threshold
        self._favor = favor

        self._constraints = list(constraints)
        self._mode = mode  # min or max
        self._state = None  # 'value', 'gain', 'offer', 'answer?' or 'go?'
        #  according to what the agent is currently waiting for

        # some constraints might be unary, and our variable can have several
        # constraints involving the same variable
        self._neighbors = set([v for c in constraints
                               for v in c.dimensions if v != variable])
        # Agent view of its neighbors resp. for ok and improve modes
        self._neighbors_values = {}
        self._neighbors_gains = {}
        self.__nb_received_offers__ = 0
        self._offers = []
        self._potential_gain = 0  # Best gain that could be achieved by a move
        self._potential_value = None  # Value for best potentila gain
        self._can_move = False

    def footprint(self):
        return computation_memory(self.computation_def.node)

    @property
    def utilities(self):
        return self._constraints

    @property
    def neighbors(self):
        return list(self._neighbors)

    def on_start(self):
        """
        Start the computation node with randomly choosing a value for its
        variable and entering value mode.

        """
        if not self.neighbors:
            # If we don't have any neighbor, simply select the best value
            # for us and be done with it !
            vals, cost = self._compute_best_value()
            val = random.choice(vals)
            self.value_selection(val, cost)
            self.logger.info('No neighbors: stop immediately with value %s - '
                             '%s', val, cost)
            self.finished()

        else:
            # At start, we don't have any information to compute the cost,
            # simply use None
            if self.variable.initial_value is None:
                self.value_selection(random.choice(self.variable.domain), None)
                self.logger.info('%s mgm2 starts: randomly select value %s and '
                                 'send to neighbors', self.variable.name,
                                 self.current_value)
            else:
                self.value_selection(self.variable.initial_value, None)
                self.logger.info('%s mgm starts: select initial value %s and '
                                 'send to neighbors',
                                 self.variable.name, self.current_value)

            self._send_value()
            self._enter_state('value')

    def _compute_best_value(self):
        """
        Compute the best eval value that can be achieved by a unilateral move,
        wrt neighbors value.

        :return: (list of variable best values, best eval of the cost/utility)

        """
        asgt = self._neighbors_values.copy()
        best_cost, best_val = None, []

        for v in self._variable.domain:
            asgt[self.variable.name] = v
            c = self._compute_cost(asgt)
            if best_cost is None \
                    or (best_cost > c and self._mode == 'min') \
                    or (best_cost < c and self._mode == 'max'):
                best_cost = c
                best_val = [v]
            elif best_cost == c:
                best_val.append(v)

        return best_val, best_cost

    def _compute_offers_to_send(self):
        """
        Computes all the coordinated moves with the partner (if exists).
        It also set the attribute best_unilateral_move, which corresponds to
        the best eval the agent can achieve if it moves alone and the list of
        values to achieve this eval

        :return: a dictionary which keys are couples (my_value,
        my_partner_value) and which values are the gain realized by the
        offerer thanks to this coordinated change.

        """
        partial_asgt = self._neighbors_values.copy()
        offers = dict()

        for limited_asgt in generate_assignment_as_dict([self.variable,
                                                         self._partner]):
            partial_asgt.update(limited_asgt)
            cost = self._compute_cost(partial_asgt)

            if (self.current_cost > cost and self._mode == 'min') or \
                    (self.current_cost < cost and self._mode == 'max'):
                offers[(limited_asgt[self.name],
                        limited_asgt[self._partner.name])] = self.current_cost\
                                                             - cost
        return offers

    def _find_best_offer(self, all_offers):
        """
        Find the offer that maximize the global gain of both partners in
        the given offers and for the given partner.
        :param all_offers: a list of couples (offerer_name, offer) where
        offer is a dictionary of offers {(partner_val, my_val): partner_gain}
        Mgm2OfferMessage
        :return: (list of best offers, global_gain)
        """
        bests, best_gain = [], 0

        for partner, offers in all_offers:
            partial_asgt = self._neighbors_values.copy()
            current_partner = self._neighbor_var(partner)

            # Filter out the constraints linking those two variables to avoid
            # counting their cost twice.
            shared = find_dependent_relations(current_partner,
                                              self._constraints)
            concerned = [rel for rel in self._constraints if rel not in shared]

            for (val_p, my_offer_val), partner_local_gain in offers.items():
                partial_asgt.update({partner: val_p,
                                     self.variable.name: my_offer_val})

                # Then we evaluate the agent constraint's for the offer
                # and add the partner's local gain.
                cost = self._compute_cost(partial_asgt, concerned)
                global_gain = self.current_cost - cost + partner_local_gain

                if (global_gain > best_gain and self._mode == 'min') \
                        or (global_gain < best_gain and self._mode == 'max'):
                    bests = [(val_p, my_offer_val, partner)]
                    best_gain = global_gain
                elif global_gain == best_gain:
                    bests.append((val_p, my_offer_val, partner))

        return bests, best_gain

    def _send_value(self):
        """
        Send a Mgm2ValueMessage to the neighbors to inform them of the variable
        current value

        """
        self.new_cycle()
        if self.cycle_stop is not None and self.cycle_count >= self.cycle_stop:
            # The computation has run for the requested number of cycles :
            # stop it.
            self.logger.info('Computation has reached the number of '
                             'requested cycles (%s) : stopping ',
                             self.cycle_stop)
            self.finished()
            return
        else:
            self.logger.debug('new cycle %s', self.cycle_count)

        msg = Mgm2ValueMessage(self.current_value)
        self.logger.debug('%s sends value message %s to %s', self.name, msg,
                          [n.name for n in self.neighbors])
        for n in self.neighbors:
            self.post_msg(n.name, msg)

    def _send_offer(self, real_offer):
        """
        Send an Mgm2OfferMessage to the chosen partner after computing the
        acceptable offers from the local constraints/utility function.

        :param real_offer: True if the agent is sending a true offer. False
        if it sends a "fake" offer message; which serves for synchronicity
        purpose: informing the neighbor that this agent does not send him an
        offer, so that each agent knows when it has received all the offers
        he should consider. This prevent an agent from moving forward while
        he should have waited for another offer to consider.
        """
        offers = None
        # Send offers to the partner if I'm an offerer
        if real_offer:
            offers = self._compute_offers_to_send()
            msg = Mgm2OfferMessage(offers, True)
            self.logger.debug('%s sends offer message %s to %s', self.name,
                              msg, self._partner.name)
            self.post_msg(self._partner.name, msg)

        # Inform other neighbors that it doesn't send offers to them
        for n in self.neighbors:
            if n != self._partner:
                self.post_msg(n.name, Mgm2OfferMessage(dict(), False))
            self.logger.debug('%s sends offer message %s to %s', self.name,
                              Mgm2OfferMessage(dict(), False), n)
        return offers

    def _send_gain(self):
        """
        Send a Mgm2GainMessage to neighbors to inform them of the best gain
         that the variable can achieve

        """
        self.logger.info('%s sends gain message %s to %s', self.name,
                         self._potential_gain,
                         [n.name for n in self.neighbors])
        for n in self._neighbors:
                self.post_msg(n.name, Mgm2GainMessage(self._potential_gain))

    def on_message(self, sender_name, msg, t):
        msg_state = msg.type
        if msg_state == self._state:
            self.states[msg_state](sender_name, msg)
        else:
            self.logger.debug('%s postponed message from %s for state %s : '
                              '%s ', self.variable.name, sender_name,
                              msg_state, msg)
            self._postponed_msg[msg_state].append((sender_name, msg))

    def _handle_value_message(self, variable_name, recv_msg):
        self.logger.debug('%s processes %s from %s', self.name, recv_msg,
                          variable_name)
        self._neighbors_values[variable_name] = recv_msg.value

        # Once we have a value for all neighbors:
        if len(self._neighbors_values) == len(self._neighbors):

            self.logger.debug('%s received values from all neighbors : %s',
                              self.name, self._neighbors_values)

            # We have our neighbors value , we can compute our real local cost
            self.__cost__ = self._current_local_cost()

            # random offerer choice
            if random.uniform(0, 1) < self._threshold:
                self._is_offerer = True
                self._partner = random.choice(list(self._neighbors))
                offers = self._send_offer(True)
                self.logger.info('%s is an offerer and chose %s as '
                                 'partner, offers: %s', self.name,
                                 self._partner.name, offers)
            else:
                # Informing neighbors they won't receive an offer from me
                self.logger.info('%s is NOT an offerer ', self.name)
                self._send_offer(False)

            # Compute best unilateral move:
            best_vals, best_cost = self._compute_best_value()
            self._potential_gain = self.current_cost - best_cost

            if (self._mode == 'min' and self._potential_gain > 0) \
                    or (self._mode == 'max' and self._potential_gain < 0):
                self._potential_value = random.choice(best_vals)
            else:
                self._potential_value = self.current_value

            self._enter_state('offer')

        else:
            # Still waiting for other neighbors
            missing = set(n.name for n in self._neighbors) - \
                      set(self._neighbors_values)
            self.logger.debug('%s waiting for values from other neighbors '
                              '(missing %s, got %s,)',
                              self.name, missing,
                              [n for n in self._neighbors_values])

    def _handle_offer_msg(self, variable_name, recv_msg):
        self.logger.debug('%s processes %s from %s', self.name, recv_msg,
                          variable_name)
        self.__nb_received_offers__ += 1

        if recv_msg.is_offering:
            if self._is_offerer:
                self.post_msg(variable_name, Mgm2ResponseMessage(False))
                self.logger.info('%s refuses offer from %s (already an '
                                 'offerer)', self.name, variable_name)
            else:
                self._offers.append((variable_name, recv_msg.offers))

        # When sure that all offers have been received
        if self.__nb_received_offers__ == len(self._neighbors):
            self.logger.info("%s has all offer msg ", self.name)

            # accept the best offer if any
            best_offers, gain = self._find_best_offer(self._offers)
            if gain == 0 or not best_offers or\
                    (self._mode == 'min' and gain < self._potential_gain) or\
                    (self._mode == 'max' and gain > self._potential_gain):
                self.logger.info("%s has considered no offer as "
                                 "acceptable", self.name)
            elif (self._mode == 'min' and gain > self._potential_gain) or\
                    (self._mode == 'max' and gain < self._potential_gain):
                self.accept_offer(best_offers, gain)
            elif gain == self._potential_gain:
                if self._favor == 'coordinated':
                    self.accept_offer(best_offers, gain)
                elif self._favor == 'no':
                    if random.uniform(0, 1) > 0.5:
                        self.accept_offer(best_offers, gain)

            # send reject messages to all other offerers
            for n, _ in self._offers:
                if self._partner is not None and n == self._partner.name:
                    continue
                self.logger.info('%s refuses offer from %s',
                                 self.name, n)
                self.post_msg(n, Mgm2ResponseMessage(False))

            if self._is_offerer:
                self._enter_state('answer?')
            else:
                self._send_gain()
                self._enter_state('gain')
        else:
            self.logger.info('%s waits for other neighbors offers (got %d '
                             'messages)', self.name,
                             self.__nb_received_offers__)

    def _handle_response_msg(self, variable_name, msg: Mgm2ResponseMessage):
        # We should get a single response message, as we made a single offer.
        self.logger.debug('%s processes %s from %s', self.name,
                          msg, variable_name)
        if variable_name != self._partner.name:
            raise ValueError(
                "{} Received offer answer from {} while its partner is "
                "{} : {}".format(self.name, variable_name, self._partner, msg))
        if not self._is_offerer:
            raise ValueError(
                "{} received offer answer from {} even though it is not "
                "an offerer".format(self.name, variable_name))

        if msg.accept:
            self._potential_value = msg.value
            self._potential_gain = msg.gain
            self._committed = True
            self.logger.info('Commit to value %s due to offer from %s, '
                             'gain %s', msg.value, variable_name, msg.gain)
        else:
            self._committed = False
            self.logger.info('Offer refused, %s received reject message from '
                             '%s', self.name, variable_name)
        self._send_gain()
        self._enter_state('gain')

    def _handle_gain_message(self, variable_name, recv_msg):
        self.logger.debug('%s processes %s from %s', self.name, recv_msg,
                          variable_name)
        # TODO : only keep max gain ?
        self._neighbors_gains[variable_name] = recv_msg.value

        # if messages received from all neighbors
        if len(self._neighbors_gains) == len(self._neighbors):
            # determine if can change value and send ok message to neighbors
            if self._potential_gain == 0:
                self.logger.info('Potential gain for %s is 0: no reason to '
                                 'change local value', self.name)
                self._clear_agent()
                self._send_value()
                self._enter_state('value')
                return

            self.logger.info('%s received gain from all neighbors %s',
                             self.name, self._neighbors_gains)
            if self._committed:
                neigh_gains = [val for n, val in self._neighbors_gains.items()
                               if n != self._partner.name]
                if neigh_gains == [] or self._potential_gain > max(neigh_gains):
                    self.logger.info('%s is commited and best gain : GO for '
                                     'cordinated change with %s', self.name,
                                     self._partner.name)
                    self._can_move = True
                    self.post_msg(self._partner.name, Mgm2GoMessage(True))
                else:
                    self.logger.info('%s is commited but lower gain: NO-GO '
                                     'for cordinated change with %s',
                                     self.name, self._partner.name)
                    self._can_move = False
                    self.post_msg(self._partner.name, Mgm2GoMessage(False))
                self._enter_state('go?')

            else:
                max_neighbors = max(list(self._neighbors_gains.values()))
                if self._potential_gain > max_neighbors:
                    self.logger.info('Local gain is best, %s unilaterally '
                                     'changes its  value to %s',
                                     self.name, self._potential_value)
                    self.value_selection(self._potential_value,
                                         self.current_cost -
                                         self._potential_gain)

                elif self._potential_gain == max_neighbors:
                    ties = sorted([k for k, v in self._neighbors_gains.items()
                                   if v == max_neighbors] + [self.name])
                    if ties[0] == self.name:
                        self.logger.info(
                            ' %s won tie-break on gain %s with variable '
                            'order: %s', self.name, max_neighbors, ties)
                        self.value_selection(self._potential_value,
                                             self.current_cost -
                                             self._potential_gain)
                    else:
                        self.logger.info(
                            ' %s lost tie-break on gain %s with variable '
                            'order: %s', self.name, max_neighbors, ties)

                else:
                    self.logger.info('Lower local gain on %s: do NOT change '
                                     'value', self.name)
                self._clear_agent()
                self._send_value()
                self._enter_state('value')

        else:
            # Still waiting for other neighbors
            self.logger.debug('%s waiting for gain msg from other neighbors ('
                              'got %s)', self.name,
                              [n for n in self._neighbors_gains])

    def _handle_go_message(self, variable, msg: Mgm2GoMessage):
        self.logger.info('%s processes %s', self.name, msg)
        if msg.go:
            if self._can_move:
                self.logger.info('%s change value to %s on go message from %s',
                                 self.name, self._potential_value, variable)

                self.value_selection(self._potential_value,
                                     self.current_cost - self._potential_gain)
            else:
                self.logger.warning('%s received GO from %s, but CANNOT '
                                    'change value : another neighbor has a '
                                    'better gain than the offer global gain',
                                    self.name,
                                    variable)
        else:
            self.logger.info('%s received NO-GO from %s, do NOT change value',
                             self.name, variable)
        # End of the cycle. Resetting view & computation attributes before
        # going to next cycle
        self._clear_agent()
        self._send_value()
        self._enter_state('value')

    def _enter_state(self, state):
        self.logger.info(' %s enters state %s', self.name, state)
        self._state = state
        while self._postponed_msg[state]:
            msg = self._postponed_msg[state].pop()
            self.states[state](*msg)

    def _clear_agent(self):
        self._neighbors_values.clear()
        self._neighbors_gains.clear()
        self._offers.clear()
        self._partner = None
        self._committed = False
        self._is_offerer = False
        self._potential_gain = 0
        self._potential_value = None
        self.__nb_received_offers__ = 0
        self._can_move = False

    def _current_local_cost(self):
        assignment = self._neighbors_values.copy()
        assignment.update({self.variable.name: self.current_value})
        return self._compute_cost(assignment)

    def _compute_cost(self, assignment, constraints=None):
        constraints = self._constraints if constraints is None else constraints
        # Cost for constraints:
        cost = fp.reduce(op.add,
                         [f(**filter_assignment_dict(assignment, f.dimensions))
                          for f in constraints],
                         0)
        # Cost for variable, if any:
        concerned_vars = set(v for c in constraints for v in c.dimensions)
        for v in concerned_vars:
            if hasattr(v, 'cost_for_val'):
                cost += v.cost_for_val(assignment[v.name])

        return cost

    def _neighbor_var(self, name):
        """
        Return the variable object for the neighbor named `name`.
        :param name:
        :return:
        """
        return next(n for n in self._neighbors if n.name == name)

    def accept_offer(self, best_offers, gain):
        val_p, my_offer_val, partner_name = random.choice(
            best_offers)
        self.logger.info('%s accepts offer (%s, %s) from %s with '
                         'gain %s ', self.name, val_p, my_offer_val,
                         partner_name, gain)
        self._potential_value = my_offer_val
        self._potential_gain = gain
        self._partner = self._neighbor_var(partner_name)
        self._committed = True
        self.post_msg(partner_name, Mgm2ResponseMessage(True, val_p, gain))

    def __str__(self):
        return 'Mgm2(' + self.name + ')'

    def __repr__(self):
        return self.__str__()
