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
MGM2 : a 2-coordinated DCOP algorithm
-------------------------------------

Mgm2 algorithm as described in
'Distributed  Algorithms for DCOP: A Graphical-Game-Base Approach' (R. Maheswaran,
J. Pearce, M. Tambe, 2004)

"""
import logging
import random

from collections import defaultdict
from functools import lru_cache
from typing import Dict, Any, Tuple, List

from pydcop.algorithms import AlgoParameterDef, ComputationDef
from pydcop.infrastructure.computations import Message, VariableComputation, register

from pydcop.computations_graph.constraints_hypergraph import VariableComputationNode
from pydcop.dcop.relations import (
    find_dependent_relations,
    generate_assignment_as_dict,
    assignment_cost,
    optimal_cost_value,
)

__author__ = "Pierre Nagellen, Pierre Rust"

GRAPH_TYPE = "constraints_hypergraph"

HEADER_SIZE = 100
UNIT_SIZE = 5


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
    neighbors = set(
        (n for l in computation.links for n in l.nodes if n not in computation.name)
    )
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
        raise ValueError(
            "target variable {} not found in constraints for {}".format(target, src)
        )

    nb_pairs = len(target_v.domain) * len(src.variable.domain)

    # for potential coordinated move we have two value and a gain :
    return nb_pairs * UNIT_SIZE * 3 + HEADER_SIZE


# Algorithm's parameters:
# ----------------------
# threshold: float
#     the threshold under which the agent is an offerer. This must be
#     between 0 and 1.
# favor: 'unilateral',
#     the type of moved that is favored in the algorithm : 'unilateral', 'no'
#     or 'coordinated'
# stop_cycle: int
#     number of cycles before stopping. If None, the computation does not
#     stop autonomously.

algo_params = [
    AlgoParameterDef("threshold", "float", None, 0.5),
    AlgoParameterDef("favor", "str", ["unilateral", "no", "coordinated"], "unilateral"),
    AlgoParameterDef("stop_cycle", "int", None, 0),
]


# ############################   MESSAGES   ################################
class Mgm2ValueMessage(Message):
    """
    Class to send a message informing neighbors of the agent value

    """

    def __init__(self, value):
        super().__init__("value", None)
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return 1

    def __str__(self):
        return "Mgm2ValueMessage({})".format(self.value)

    def __repr__(self):
        return "Mgm2ValueMessage({})".format(self.value)

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
        super().__init__("gain", None)
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return 1

    def __str__(self):
        return "Mgm2GainMessage({})".format(self.value)

    def __repr__(self):
        return "Mgm2GainMessage({})".format(self.value)

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

    def __init__(self, offers: Dict[Tuple[Any, Any], float] = None, is_offering=False):
        super().__init__("offer", None)
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
        return max(1, 3 * len(self._offers))

    def _simple_repr(self):
        r = {
            "__module__": self.__module__,
            "__qualname__": self.__class__.__qualname__,
            "is_offering": self.is_offering,
            "var_values": list(),
            "gains": list(),
        }

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
                r["var_values"] = var_values
                r["gains"] = gains

        return r

    @classmethod
    def _from_repr(cls, r):
        if "gains" in r:
            var_values = [tuple(couple) for couple in r["var_values"]]
            gains = r["gains"]

            return Mgm2OfferMessage(dict(zip(var_values, gains)), r["is_offering"])

        return Mgm2OfferMessage(dict(), r["is_offering"])

    def __str__(self):
        return "Mgm2OfferMessage({},{})".format(self.is_offering, self.offers)

    def __repr__(self):
        return "Mgm2OfferMessage({},{})".format(self.is_offering, self.offers)

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
        super().__init__("answer?", None)

        self._accept = accept
        if accept:
            if (value is None) or (gain is None):
                raise ValueError(
                    "If you send an accept message, you must send"
                    "the neighbor value and the global gain in "
                    "it too"
                )
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
        return "Mgm2ResponseMessage({},{})".format(self.accept, self.value, self._gain)

    def __repr__(self):
        return "Mgm2ResponseMessage({},{})".format(self.accept, self.value, self._gain)

    def __eq__(self, other):
        if type(other) != Mgm2ResponseMessage:
            return False
        if (
            self.accept == other.accept
            and self.value == other.value
            and self.gain == other.gain
        ):
            return True
        return False


class Mgm2GoMessage(Message):
    """
    Class to send my commited partner if we can change our values or not

    """

    def __init__(self, go: bool):
        super().__init__("go?", None)
        self._go = go

    @property
    def go(self):
        return self._go

    @property
    def size(self):
        return 1

    def __str__(self):
        return "Mgm2GoMessage({})".format(self.go)

    def __repr__(self):
        return "Mgm2GoMessage({})".format(self.go)

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

    computation_def: ComputationDef
        The computation definition this computation has been built from.

    """

    def __init__(self, computation_def: ComputationDef = None):
        assert computation_def.algo.algo == "mgm2"
        super().__init__(computation_def.node.variable, computation_def)

        self._mode = computation_def.algo.mode
        self.stop_cycle = computation_def.algo.param_value("stop_cycle")
        self._threshold = computation_def.algo.param_value("threshold")
        self._favor = computation_def.algo.param_value("favor")

        # Handling messages arriving during wrong mode
        self._postponed_msg = defaultdict(lambda: [])  # type: Dict[str, List]

        self._partner = None
        self._committed = False
        self._is_offerer = False

        self._constraints = list(computation_def.node.constraints)
        self._state = None  # 'value', 'gain', 'offer', 'answer?' or 'go?'
        #  according to what the agent is currently waiting for

        # some constraints might be unary, and our variable can have several
        # constraints involving the same variable
        self._neighbors = set(
            [v for c in self._constraints for v in c.dimensions if v != self.variable]
        )
        # Agent view of its neighbors resp. for ok and improve modes
        self._neighbors_values = {}
        self._neighbors_gains = {}
        self.__nb_received_offers__ = 0
        self._offers = []
        self._potential_gain = 0  # Best gain that could be achieved by a move
        self._potential_value = None  # Value for best potentila gain
        self._can_move = False

    @property
    def utilities(self):
        return self._constraints

    @property
    def neighbors_vars(self):
        return list(self._neighbors)

    def on_start(self):
        """
        Start the computation node with randomly choosing a value for its
        variable and entering value mode.

        """
        if not self.neighbors_vars:
            # If we don't have any neighbor, simply select the best value
            # for us and be done with it !
            vals, cost = self._compute_best_value()
            value = random.choice(vals)
            self.value_selection(value, cost)
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"No neighbors: stop immediately with value {value} - {cost}"
                )
            self.finished()

        else:
            # At start, we don't have any information to compute the cost,
            # simply use None
            if self.variable.initial_value is None:
                self.value_selection(random.choice(self.variable.domain), None)
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"Select random initial value {self.current_value} "
                    )

            else:
                self.value_selection(self.variable.initial_value, None)
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(f"Select initial value {self.current_value}")

            self._send_value()
            self._enter_state("value")

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
            c = self._compute_cost(**asgt)
            if (
                best_cost is None
                or (best_cost > c and self._mode == "min")
                or (best_cost < c and self._mode == "max")
            ):
                best_cost = c
                best_val = [v]
            elif best_cost == c:
                best_val.append(v)

        return best_val, best_cost

    def _compute_offers_to_send(self) -> Dict[Tuple[float, float], float]:
        """
        Computes all the coordinated moves with the partner (if exists).
        It also set the attribute best_unilateral_move, which corresponds to
        the best eval the agent can achieve if it moves alone and the list of
        values to achieve this eval

        Returns
        -------
        offers:
            a dictionary which keys are couples (my_value, my_partner_value)
            and which values are the gain realized by the offerer thanks to this
            coordinated change.

        """
        partial_asgt = self._neighbors_values.copy()
        offers = dict()

        for limited_asgt in generate_assignment_as_dict([self.variable, self._partner]):
            partial_asgt.update(limited_asgt)
            cost = self._compute_cost(**partial_asgt)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"looking for offer : {partial_asgt} - cost {cost}"
                    f" current {self.current_cost} {self._mode}"
                )

            if (self.current_cost > cost and self._mode == "min") or (
                self.current_cost < cost and self._mode == "max"
            ):
                offers[(limited_asgt[self.name], limited_asgt[self._partner.name])] = (
                    self.current_cost - cost
                )
        return offers

    def _find_best_offer(
        self, all_offers: List[Tuple[str, Dict]]
    ) -> Tuple[List, float]:
        """
        Find the offer that maximize the global gain of both partners in
        the given offers and for the given partner.

        Parameters
        ----------
        all_offers: list
            a list of couples (offerer_name, offer) where offer is a dictionary of
            offers {(partner_val, my_val): partner_gain} Mgm2OfferMessage

        Returns
        -------
        list:
            list of best offers (i.ee with the best gain)
        best_gain:
            gain for the best offers.
        """
        bests, best_gain = [], 0

        for partner, offers in all_offers:
            partial_asgt = self._neighbors_values.copy()
            current_partner = self._neighbor_var(partner)

            # Filter out the constraints linking those two variables to avoid
            # counting their cost twice.
            shared = find_dependent_relations(current_partner, self._constraints)
            concerned = [rel for rel in self._constraints if rel not in shared]

            for (val_p, my_offer_val), partner_local_gain in offers.items():
                partial_asgt.update({partner: val_p, self.variable.name: my_offer_val})

                # Then we evaluate the agent constraint's for the offer
                # and add the partner's local gain.
                cost = assignment_cost(partial_asgt, concerned)
                global_gain = self.current_cost - cost + partner_local_gain

                if (global_gain > best_gain and self._mode == "min") or (
                    global_gain < best_gain and self._mode == "max"
                ):
                    bests = [(val_p, my_offer_val, partner)]
                    best_gain = global_gain
                elif global_gain == best_gain:
                    bests.append((val_p, my_offer_val, partner))

        return bests, best_gain

    def _send_value(self):
        """
        Send the current value to neighbors.

        At the same time, check if the computation should be stopped.

        """
        self.new_cycle()
        if self.stop_cycle and self.cycle_count >= self.stop_cycle:
            # The computation has run for the requested number of cycles :
            # stop it.
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"Computation has reached the number of "
                    f"requested cycles ({self.stop_cycle}) : stopping "
                )
            self.finished()
            return
        else:
            self.logger.debug("new cycle %s", self.cycle_count)

        msg = Mgm2ValueMessage(self.current_value)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"{self.name} sends value message {msg} "
                f"to {[n.name for n in self.neighbors_vars]}"
            )
        for n in self.neighbors_vars:
            self.post_msg(n.name, msg)

    def on_stop(self):
        super().on_stop()

    def _send_gain(self):
        """
        Send a Mgm2GainMessage to neighbors to inform them of the best gain
         that the variable can achieve

        """
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                f"{self.name} sends gain message {self._potential_gain} "
                f"to {[n.name for n in self.neighbors_vars]}"
            )
        for n in self._neighbors:
            self.post_msg(n.name, Mgm2GainMessage(self._potential_gain))

    @register("value")
    def on_value_msg(self, sender_name, msg, t):
        if "value" == self._state:
            self._neighbors_values[sender_name] = msg.value
            if len(self._neighbors_values) == len(self._neighbors):

                self._handle_value_messages()
            else:
                if self.logger.isEnabledFor(logging.DEBUG):
                    missing = set(n.name for n in self._neighbors) - set(
                        self._neighbors_values
                    )
                    self.logger.debug(
                        f"Waiting for values from other neighbors (missing {missing},"
                        f" got {[n for n in self._neighbors_values]})"
                    )
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"{self.variable.name} postponed message from {sender_name} "
                    f"for value : {msg} "
                )
            self._postponed_msg["value"].append((sender_name, msg, t))

    @register("gain")
    def on_gain_msg(self, sender_name, msg, t):
        if "gain" == self._state:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"processes {msg} from {sender_name}")
            self._neighbors_gains[sender_name] = msg.value

            # if messages received from all neighbors
            if len(self._neighbors_gains) == len(self._neighbors):

                self._handle_gain_messages()
            else:
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info("Waiting for other neighbors gains")
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"{self.variable.name} postponed message from {sender_name} "
                    f"for gain : {msg} "
                )
            self._postponed_msg["gain"].append((sender_name, msg, t))

    @register("offer")
    def on_offer_msg(self, sender_name, msg, t):
        if "offer" == self._state:
            self._offers.append((sender_name, msg))
            # When sure that all offers have been received
            if len(self._offers) == len(self._neighbors):

                self._handle_offer_messages()
            else:
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info("Waiting for other neighbors offers ")
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"{self.variable.name} postponed message from {sender_name} "
                    f"for offer : {msg} "
                )
            self._postponed_msg["offer"].append((sender_name, msg, t))

    @register("answer?")
    def on_answer_msg(self, sender_name, msg, t):
        if "answer?" == self._state:

            self._handle_response_message(sender_name, msg)
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"{self.variable.name} postponed message from {sender_name} "
                    f"for answer? : {msg} "
                )
            self._postponed_msg["answer?"].append((sender_name, msg, t))

    @register("go?")
    def on_go_msg(self, sender_name, msg, t):
        if "go?" == self._state:
            self._handle_go_message(sender_name, msg)
        else:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"{self.variable.name} postponed message from {sender_name} "
                    f"for go? : {msg} "
                )
            self._postponed_msg["go?"].append((sender_name, msg, t))

    def _handle_value_messages(self):

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"{self.name} received values from all neighbors : {self._neighbors_values}"
            )

        # We have our neighbors value , we can compute our real local cost
        self.__cost__ = self._current_local_cost()

        # random offerer choice
        self._partner = None
        self._is_offerer = False
        if random.uniform(0, 1) < self._threshold:
            self._is_offerer = True
            self._partner = random.choice(list(self._neighbors))
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"{self.name} is an offerer and chose {self._partner.name} "
                    f"as partner"
                )

        for n in self.neighbors_vars:
            if n != self._partner:
                self.post_msg(n.name, Mgm2OfferMessage(dict(), False))
            else:
                self.post_msg(
                    n.name, Mgm2OfferMessage(self._compute_offers_to_send(), True)
                )
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"{self.name} sends offer message to {n}")

        # Compute best unilateral move:
        best_vals, best_cost = self._compute_best_value()
        self._potential_gain = self.current_cost - best_cost

        if (self._mode == "min" and self._potential_gain > 0) or (
            self._mode == "max" and self._potential_gain < 0
        ):
            self._potential_value = random.choice(best_vals)
        else:
            self._potential_value = self.current_value

        self._enter_state("offer")

    def _handle_offer_messages(self):

        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"{self.name} has all offer msg ")

        if self._is_offerer:
            for sender, offer_msg in self._offers:
                if offer_msg.is_offering:
                    self.post_msg(sender, Mgm2ResponseMessage(False))
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"Refusing offer from {sender} (already an offerer)"
                        )
            self._enter_state("answer?")

        else:
            # accept the best offer if any
            best_offers, gain = self._find_best_offer(
                [
                    (sender, offer_msg.offers)
                    for sender, offer_msg in self._offers
                    if offer_msg.is_offering
                ]
            )
            self._committed = False
            if gain == 0 or not best_offers:
                self._committed = False
            elif (self._mode == "min" and gain > self._potential_gain) or (
                self._mode == "max" and gain < self._potential_gain
            ):
                self._committed = True
            elif gain == self._potential_gain:
                if self._favor == "coordinated":
                    self._committed = True
                elif self._favor == "no" and random.uniform(0, 1) > 0.5:
                    self._committed = True

            val_p = None
            if self._committed:
                val_p, self._potential_value, partner_name = random.choice(best_offers)
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"Accepts offer ({val_p}, {self._potential_value}) "
                        f"from {partner_name} with gain {gain}"
                    )
                self._potential_gain = gain
                self._partner = self._neighbor_var(partner_name)
            else:
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(f"No accepted offer")

            # send accept / reject messages to all offerers
            for sender, offer_msg in self._offers:
                if not offer_msg.is_offering:
                    continue
                if self._is_offerer:
                    self.post_msg(sender, Mgm2ResponseMessage(False))
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"Refusing offer from {sender} (already an offerer)"
                        )
                elif self._partner and sender == self._partner.name:
                    self.post_msg(sender, Mgm2ResponseMessage(True, val_p, gain))
                else:
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(f"Refusing offer from {sender}")
                    self.post_msg(sender, Mgm2ResponseMessage(False))

            self._send_gain()
            self._enter_state("gain")

    def _handle_response_message(self, variable_name, msg: Mgm2ResponseMessage):
        # We should get a single response message, as we made a single offer.
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"processes {msg} from {variable_name}")

        if variable_name != self._partner.name:
            raise ValueError(
                f"{self.name} Received offer answer from {variable_name} while its partner "
                f"is {self._partner} : {msg}"
            )
        if not self._is_offerer:
            raise ValueError(
                f"{self.name} received offer answer from {variable_name} even though it "
                f"is not an offerer"
            )

        if msg.accept:
            self._potential_value = msg.value
            self._potential_gain = msg.gain
            self._committed = True
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"Commit to value {msg.value} due to offer from {variable_name}, "
                    f"gain {msg.gain}"
                )
        else:
            self._committed = False
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"Offer refused, received reject message from {variable_name}"
                )
        self._send_gain()
        self._enter_state("gain")

    def _handle_gain_messages(self):

        # determine if can change value and send ok message to neighbors
        if self._potential_gain == 0:
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"Potential gain is 0: no reason to change local value"
                )
            self._clear_agent()
            self._send_value()
            self._enter_state("value")
            return
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                f"Received gain from all neighbors {self._neighbors_gains}"
            )
        if self._committed:
            neigh_gains = [
                val
                for n, val in self._neighbors_gains.items()
                if n != self._partner.name
            ]
            if neigh_gains == [] or self._potential_gain > max(neigh_gains):
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"Commited and best gain : GO for "
                        f"coordinated change with {self._partner.name}"
                    )
                self._can_move = True
                self.post_msg(self._partner.name, Mgm2GoMessage(True))
            else:
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"Commited but lower gain: NO-GO "
                        f"for cordinated change with {self._partner.name}"
                    )
                self._can_move = False
                self.post_msg(self._partner.name, Mgm2GoMessage(False))
            self._enter_state("go?")

        else:
            max_neighbors = max(list(self._neighbors_gains.values()))
            if self._potential_gain > max_neighbors:
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"Local gain is best, {self.name} unilaterally changes its "
                        f"value to {self._potential_value}"
                    )
                self.value_selection(
                    self._potential_value, self.current_cost - self._potential_gain
                )

            elif self._potential_gain == max_neighbors:
                ties = sorted(
                    [k for k, v in self._neighbors_gains.items() if v == max_neighbors]
                    + [self.name]
                )
                if ties[0] == self.name:
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"Won tie-break on gain {max_neighbors} "
                            f"with variable order: {ties}"
                        )
                    self.value_selection(
                        self._potential_value, self.current_cost - self._potential_gain
                    )
                else:
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"Lost tie-break on gain {max_neighbors} "
                            f"with variable order: {ties}"
                        )

            else:
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"Lower local gain on {self.name}: do NOT change " "value"
                    )
            self._clear_agent()
            self._send_value()
            self._enter_state("value")

    def _handle_go_message(self, variable: str, msg: Mgm2GoMessage):
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"processes {msg} from {variable}")
        if msg.go:
            if self._can_move:
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        f"Change value to {self._potential_value} "
                        f"on go message from {variable}"
                    )

                self.value_selection(
                    self._potential_value, self.current_cost - self._potential_gain
                )
            else:
                if self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        f"Received GO from {variable}, but CANNOT change value: "
                        f"another neighbor has a better gain than the offer global gain"
                    )
        else:
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"Received NO-GO from {variable}, do NOT change value")
        # End of the cycle. Resetting view & computation attributes before
        # going to next cycle
        self._clear_agent()
        self._send_value()
        self._enter_state("value")

    def _enter_state(self, state):
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"Enters state {state}")
        self._state = state
        while self._postponed_msg[state]:
            msg = self._postponed_msg[state].pop()
            if state == "value":
                self.on_value_msg(*msg)
            elif state == "offer":
                self.on_offer_msg(*msg)
            elif state == "answer?":
                self.on_answer_msg(*msg)
            elif state == "gain":
                self.on_gain_msg(*msg)
            elif state == "go?":
                self.on_go_msg(*msg)
            else:
                raise ValueError(f"Unkown mgm2 state {state}")

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
        return self._compute_cost(**assignment)

    def _neighbor_var(self, name):
        """
        Return the variable object for the neighbor named `name`.
        :param name:
        :return:
        """
        return next(n for n in self._neighbors if n.name == name)

    def accept_offer(self, best_offers, gain):
        val_p, my_offer_val, partner_name = random.choice(best_offers)
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                f"Accepts offer ({val_p}, {my_offer_val}) "
                f"from {partner_name} with gain {gain}"
            )
        self._potential_value = my_offer_val
        self._potential_gain = gain
        self._partner = self._neighbor_var(partner_name)
        self._committed = True
        self.post_msg(partner_name, Mgm2ResponseMessage(True, val_p, gain))

    @lru_cache(maxsize=512)
    def _compute_cost(self, **kwargs):
        return assignment_cost(kwargs, self._constraints)
