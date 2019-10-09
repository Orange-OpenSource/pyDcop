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

Distributed algorithm for Minimum Spanning Tree (MST)


Compared to the original algorithm, an extra `terminate` message has been added
to inform all nodes in the MST that the algorithm has terminated.

Note: many comments in this implementation mention 'min' (for example in the expression
"min-weight outre edge), however it can actually be max when building a maximum
weight spanning tree.

* TODO: select node that wake up
  especially for disconnected graphs !

"""
from enum import Enum
from typing import List, Tuple

from pydcop.infrastructure.computations import (
    MessagePassingComputation,
    register,
    ComputationException,
    message_type,
)


class NodeState(Enum):
    """ States for nodes of the graph"""

    SLEEPING = 1  # The node is currently not participating in th algorithm (yet)
    FIND = 2  # Currently looking for the min/max weight outer edge for the fragment
    FOUND = 3  # the min/max weight edge has been found


class EdgeLabel(Enum):
    """ Labels for edges in the graph. """

    BASIC = 1  # When we have not decided if the edge is in the MST.
    BRANCH = 2  # The edge is in the MST.
    REJECTED = 3  # The edge is NOT in the MST.


# Messages definitions
ConnectMessage = message_type("connect", ["sender", "level"])
InitiateMessage = message_type("initiate", ["sender", "level", "id", "state"])
TestMessage = message_type("test", ["sender", "level", "id"])
AcceptMessage = message_type("accept", ["sender"])
RejectMessage = message_type("reject", ["sender"])
ReportMessage = message_type("report", ["sender", "weight"])
ChangeRootMessage = message_type("change_root", ["sender"])
ChangeRootMessage = message_type("change_root", ["sender"])
TerminateMessage = message_type("terminate", ["sender"])


class SpanningTreeComputation(MessagePassingComputation):
    """
    Computation for building a  Minimum (or maximum) weight spanning tree.

    Parameters
    ----------
    name: str
        the name of that node
    neighbors: list
        a list of tuples, which contains the name of a neighbor node with the weight of
        the edge to that neighbor.
    mode: str
        `min` or `max`
    wakeup_at_start: boolean
        Indicates if that node should wake spontaneously at startup, defaults to False.
    """

    def __init__(
        self,
        name: str,
        neighbors: List[Tuple[str, float]],
        mode: str = "min",
        wakeup_at_start=False,
    ):
        super().__init__(name)
        if mode not in ["min", "max"]:
            raise ComputationException(
                f"Invalid mode in SpanningTreeComputation, use 'min or 'max', not {mode}"
            )
        self.mode = mode

        # Weights are used as frament'id, for that reason they must be unique
        # as we also want to support weighted graph with non-distinct graphs,
        # we add the edge in our internal weight definition.
        # E.g we use (2, "A", "C), instead of simply 3,  as the weight for the edge
        # ("A", "C"). Note that the edge's nodes must be sorted !
        self.weights = {n: (w, to_edge(name, n)) for n, w in neighbors}

        # Each edge is labelled to indicated if it is part of the MST or not
        # (or if we haven't decided yet)
        self.labels = {n: EdgeLabel.BASIC for n, w in neighbors}
        self.state = NodeState.SLEEPING
        self.level = None
        self.find_count = None
        self.fragment_identity = None
        self.in_branch = None
        self.best_edge = None
        self.best_weight = None
        self.test_edge = None
        self.wakeup_at_start = wakeup_at_start

    @property
    def all_labelled(self):
        """
        A node is `all_labelled` once all its edges have been labelled as BRANCH of REJECTED.

        At the end, all node must have `all_labelled` equals to `True`.

        Note that this does not mean that the whole distributed algorithm has
        terminated, a node where `all_labelled` is True may still receive messages and
        MUST NOT be stopped.

         """
        return all(label != EdgeLabel.BASIC for label in self.labels.values())

    def on_start(self):
        # wake-up at random
        if self.wakeup_at_start:
            self.wakeup()
        # else:
        #     # FIXME: re-try if nobody wakes up after some time ?
        #     if random() > 0.5:
        #         self.wakeup()

    def wakeup(self):
        """Wakeup make a single node a fragment of level 0."""
        if self.state != NodeState.SLEEPING:
            self.logger.error(f"Cannot wake up when not sleeping at {self.name}")
            raise ComputationException("Cannot wake up when not sleeping")

        try:
            best_edge, best_weight = find_best_edge(self.weights, self.mode)
            self.logger.debug(f"Wake up on {self.name}  - send connect to {best_edge}")

            self.labels[best_edge] = EdgeLabel.BRANCH
            self.level = 0
            self.state = NodeState.FOUND
            self.find_count = 0
            self.post_msg(best_edge, ConnectMessage(self.name, 0))
            # Now we wait from a message from `best_edge`

        except ValueError:
            # this node has no neighbors ! nothing to do
            self.state = NodeState.FOUND

    @register("connect")
    def on_connect(self, _sender: str, msg: ConnectMessage, t: float) -> None:
        """
        A connect message is sent over the minimum-weight outer edge of a fragment,
        to connect to another fragment.

        Parameters
        ----------
        _sender: str
            sender of the message
        msg : ConnectMessage
            message
        t : float
            time
        """
        self.logger.debug(f"On connect on {self.name}  - {msg} at {t}")

        if self.state == NodeState.SLEEPING:
            self.wakeup()
        if msg.level < self.level:
            self.logger.debug(f"Connect from lower level fragment {msg} on {self.name}")
            self.labels[msg.sender] = EdgeLabel.BRANCH
            self.post_msg(
                msg.sender,
                InitiateMessage(
                    self.name, self.level, self.fragment_identity, self.state
                ),
            )
            if self.state == NodeState.FIND:
                self.find_count += 1
        else:
            if self.labels[msg.sender] == EdgeLabel.BASIC:
                # Postpone message until our level is higher
                self.logger.debug(f"Postponing on {self.name} msg {msg}")
                self.post_msg(self.name, msg)
            else:
                new_id = self.weights[msg.sender]
                self.logger.debug(
                    f"New fragment creation on {self.name}  "
                    f" new id: {new_id} level: {self.level+1}"
                )
                self.post_msg(
                    msg.sender,
                    InitiateMessage(self.name, self.level + 1, new_id, NodeState.FIND),
                )

    @register("initiate")
    def on_initiate(self, _sender: str, msg: InitiateMessage, _t: float):
        """
        An `initiate` message starts the search for the minimum-weight outer edge for a
        fragment.

        This search is performed by:
        * propagating `initiate` messages on all `BRANCH` edges of the fragment
        * send test messages sequentially on `BASIC` edges,
        
        Parameters
        ----------
        _sender: str
            sender of the message
        msg: InitiateMessage
            message
        _t: float
            time
        """
        self.logger.debug(f"Initiate from {msg.sender} on {self.name} : {msg}")
        self.level = msg.level
        self.fragment_identity = msg.id
        self.state = msg.state
        self.in_branch = msg.sender
        self.best_edge = None
        self.best_weight = inf_val(self.mode)

        # Send initiate to all other neighbors
        for neighbor in self.weights:
            if neighbor == msg.sender or self.labels[neighbor] != EdgeLabel.BRANCH:
                continue
            self.post_msg(
                neighbor,
                InitiateMessage(
                    self.name, self.level, self.fragment_identity, self.state
                ),
            )
            # Increase the number of neighbors in FIND state for each neighbor
            # we flip to that state, in order to be able to check later if we received
            # all their responses.
            if self.state == NodeState.FIND:
                self.find_count += 1
        if self.state == NodeState.FIND:
            self.test()

    def test(self):
        self.logger.debug(f"Test `BASIC` adjacent edges on {self.name}")

        try:
            best_edge, best_weight = find_best_edge(
                self.weights, self.mode, self.labels, EdgeLabel.BASIC
            )
            self.logger.debug(
                f"Found best weight 'BASIC' edge {best_edge}, send test on id "
            )
            self.test_edge = best_edge
            self.post_msg(
                best_edge, TestMessage(self.name, self.level, self.fragment_identity)
            )
        except ValueError:
            # No `BASIC` adjacent edge here, can report directly
            self.logger.debug(
                f"No `BASIC` adjacent edge on {self.name}, can report directly"
            )
            self.test_edge = None
            self.report()

    @register("test")
    def on_test(self, _sender, msg: TestMessage, _t: float):
        """
        A `test` message is send over a `BASIC` edge, to check if that edge is an
        outgoing edge for the fragment.
        * If the node is not in the same fragment, it accept the test,
          meaning that it is indeed an outgoing edge
        * otherwise, either the fragment id is the same (and it's not an outgoing edge)
          or the local level is not high enough and the answer is postponed.

        Parameters
        ----------
        _sender
        msg
        _t

        """
        self.logger.debug(f"On test msg on {self.name} : {msg}")
        if self.state == NodeState.SLEEPING:
            self.wakeup()

        if self.level < msg.level:
            # Postpone msg until this node reach an appropriate level
            self.logger.debug(f"Postponing on {self.name} msg {msg} {self.level}")
            self.post_msg(self.name, msg)
        else:
            if self.fragment_identity != msg.id:
                self.logger.debug(f"Accept test from {msg.sender} on {self.name}")
                self.post_msg(msg.sender, AcceptMessage(self.name))
            else:
                self.logger.debug(
                    f"test from same level {msg} on {self.name} {self.level} {self.fragment_identity}"
                )
                if self.labels[msg.sender] == EdgeLabel.BASIC:
                    self.logger.debug(
                        f"Label edge {msg.sender} - {self.name} as `REJECTED`"
                    )
                    self.labels[msg.sender] = EdgeLabel.REJECTED
                if self.test_edge != msg.sender:
                    self.logger.debug(f"Reject test from {msg.sender} on {self.name}")
                    self.post_msg(msg.sender, RejectMessage(self.name))
                else:
                    self.logger.debug(f"Try test on next edge on {self.name}")
                    self.test()

    @register("reject")
    def on_reject(self, _sender, msg, _t: float):
        self.logger.debug(f"On reject msg from {msg.sender} on {self.name} : {msg}")
        if self.labels[msg.sender] == EdgeLabel.BASIC:
            self.labels[msg.sender] = EdgeLabel.REJECTED
        self.test()

    @register("accept")
    def on_accept(self, _sender, msg, _t: float):
        self.logger.debug(f"On accept msg from {msg.sender} on {self.name} : {msg}")
        self.test_edge = None
        if is_best_weight(self.weights[msg.sender], self.best_weight, self.mode):
            self.best_edge = msg.sender
            self.best_weight = self.weights[msg.sender]
        self.report()

    def report(self):
        self.logger.debug(
            f"Report {self.name} find {self.find_count}  - {self.test_edge}"
        )
        if self.find_count == 0 and self.test_edge is None:
            self.state = NodeState.FOUND
            self.post_msg(self.in_branch, ReportMessage(self.name, self.best_weight))

    @register("report")
    def on_report(self, _sender, msg: ReportMessage, _t: float):
        self.logger.debug(
            f"On report msg from {msg.sender} on {self.name} : {msg} {self.state}"
        )

        if msg.sender != self.in_branch:
            self.find_count -= 1
            if is_best_weight(msg.weight, self.best_weight, self.mode):
                self.logger.debug(
                    f"found better weight on {self.name}, keep reporting {msg}"
                )
                self.best_weight = msg.weight
                self.best_edge = msg.sender
            self.report()
        else:
            if self.state == NodeState.FIND:
                # Postpone message:
                self.logger.debug(f"Postponing on {self.name} msg {msg}")
                self.post_msg(self.name, msg)
            else:
                if is_best_weight(self.best_weight, msg.weight, self.mode):
                    self.logger.debug(f"Changing root on {self.name} msg {msg}")
                    self.change_root()
                else:
                    if msg.weight == self.best_weight and msg.weight == inf_val(
                        self.mode
                    ):
                        self.logger.info(f"Finished on  {self.name} msg {msg}")
                        # Propagate termination info to all other nodes of the MSt
                        self.propagate_termination(msg.sender)
                        self.stop()
                    else:
                        self.logger.debug(
                            f"Cannot finish on {self.name} {msg.weight} != {self.best_weight} "
                        )

    def propagate_termination(self, origin_node):
        for node, label in self.labels.items():
            if origin_node == node:
                continue
            if label == EdgeLabel.BRANCH:
                self.post_msg(node, TerminateMessage(self.name))

    @register("terminate")
    def on_terminate(self, _sender, msg, _t):
        self.logger.debug(
            f"Received termination from {msg.sender} on {self.name}, propagate and stop"
        )
        if not self.all_labelled:
            self.logger.critical(
                f"Invalid terminate message, all edges are not labelled on {self.name} : {self.labels}"
            )
            raise ComputationException(
                f"Invalid terminate message, all edges are not labelled on {self.name} : {self.labels}"
            )
        self.propagate_termination(msg.sender)
        self.stop()

    def change_root(self):
        if self.labels[self.best_edge] == EdgeLabel.BRANCH:
            self.post_msg(self.best_edge, ChangeRootMessage(self.name))
        else:
            self.post_msg(self.best_edge, ConnectMessage(self.name, self.level))
            self.labels[self.best_edge] = EdgeLabel.BRANCH

    @register("change_root")
    def on_change_root(self, _sender: str, msg: ChangeRootMessage, _t: float):
        """

        Parameters
        ----------
        _sender : str
        msg: ChangeRootMessage
            message
        _t: float
            time
        """
        self.logger.debug(
            f"On test change_root msg from {msg.sender} on {self.name} : {msg}"
        )
        self.change_root()


def inf_val(mode: str) -> Tuple:
    """
    Return the appropriate infinite value depending on mode.

    Note that the value is returned as a single-value tuple, in order to be
    compared with weights, which are represented as tuples.

    Parameters
    ----------
    mode: str
        `min` or `max`

    Returns
    -------
    tuple
        a single-value tuple containing `-inf` or `inf`
     """
    return (float("inf"),) if mode == "min" else (-float("inf"),)


def is_best_weight(w1, w2, mode: str) -> bool:
    """
    returns True if w1 is 'better' than 'w2', depending on the mode.

    Parameters
    ----------
    w1: number
        a node weight
    w2: number
        a node weight
    mode: str
        'min' or 'max'

    Returns
    -------
    bool:
        True if w1 is 'better' than 'w2', depending on the mode.
    """
    if mode == "min":
        return w1 < w2
    else:
        return w1 > w2


def find_best_edge(
    weights, mode: str, labels=None, filter_label=None
) -> Tuple[str, float]:
    """
    Find the edge with the best weight, according to `mode`.

    Parameters
    ----------
    weights : dict
        Dict of weights.
    mode: str
        `min` or `max`
    labels: dict
        Dict of labels, only used when filtering.
    filter_label: EdgeLabel
        When used, only edges with this label will be considered.

    Returns
    -------
    tuple:
        edge name, weight

    Raises
    ------
    ValueError
        If there is no edge or the filter excluded all edges.
    """
    if filter_label is None:
        edges = [(w, e) for e, w in weights.items()]
    else:
        edges = [(w, e) for e, w in weights.items() if labels[e] == filter_label]

    if mode == "min":
        min_w, e = min(edges)
        return e, min_w
    else:
        max_w, e = max(edges)
    return e, max_w


def to_edge(node1, node2):
    """
    Make sure tuple representing edge are always ordered.
    """
    if node1 < node2:
        return node1, node2
    return node2, node1
