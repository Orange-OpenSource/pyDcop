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

Distributed algorithm for Minimum Spanning Tree

"""
from enum import Enum
from random import random
from typing import List, Tuple

from pydcop.infrastructure.computations import (
    MessagePassingComputation,
    register,
    ComputationException,
    message_type,
)


class State(Enum):
    SLEEPING = 1
    FIND = 2
    FOUND = 3
    DONE = 4


class EdgeLabel(Enum):
    BASIC = 1
    BRANCH = 2
    REJECTED = 3


ConnectMessage = message_type("connect", ["sender", "level"])
InitiateMessage = message_type("initiate", ["sender", "level", "id", "state"])
TestMessage = message_type("test", ["sender", "level", "id"])
AcceptMessage = message_type("accept", ["sender"])
RejectMessage = message_type("reject", ["sender"])
ReportMessage = message_type("report", ["sender", "weight"])
ChangeRootMessage = message_type("changeroot", ["sender"])


class SpanningTreeComputation(MessagePassingComputation):
    """
    Computation for building a  Minimum (or maximum) weight spanning tree.


    """

    def __init__(
        self,
        name: str,
        neighbors: List[Tuple[str, float]],
        mode: str = "min",
        wakeup_at_start=False,
    ):
        super().__init__(name)
        self.mode = mode
        self.neighbors_weights = {n: w for n, w in neighbors}
        self.neighbors_labels = {n: EdgeLabel.BASIC for n, w in neighbors}
        self.state = State.SLEEPING
        self.level = None
        self.find_count = None
        self.fragment_identity = None  # FIXME: id is weight and edge in our case
        self.in_branch = None
        self.best_edge = None
        self.best_weight = None
        self.test_edge = None
        self.wakeup_at_start = wakeup_at_start

    @property
    def is_done(self):
        return all(label != EdgeLabel.BASIC for label in self.neighbors_labels.values())

    def on_start(self):
        # wake-up at random
        if self.wakeup_at_start:
            self.wakeup()
        # else:
        #     # FIXME: re-try if nobody wakes up after some time ?
        #     if random() > 0.5:
        #         self.wakeup()

    def wakeup(self):
        if self.state != State.SLEEPING:
            self.logger.error(f"Cannot wake up when not sleeping at {self.name}")
            raise ComputationException("Cannot wake up when not sleeping")
        # find best out edge
        if self.mode == "min":
            best_w, best_edge = min((w, n) for n, w in self.neighbors_weights.items())

        else:
            best_w, best_edge = min((w, n) for n, w in self.neighbors_weights.items())

        self.logger.debug(f"Wake up on {self.name}  - send connect to {best_edge}")

        # if bests
        # try:
        #     best_w, best_edge = newt(bests)
        # except
        self.neighbors_labels[best_edge] = EdgeLabel.BRANCH
        self.level = 0
        self.state = State.FOUND
        self.find_count = 0
        self.post_msg(best_edge, ConnectMessage(self.name, 0))
        # Now we wait from a message from `best_edge`

    @register("connect")
    def on_connect(self, sender: str, msg: ConnectMessage, t: float):
        """
        A connect message is sent over the minimum-weight outer edge of a fragment,
        to connect to another fragment.

        Parameters
        ----------
        sender
        msg
        t

        Returns
        -------

        """
        self.logger.debug(f"On connect on {self.name}  - {msg}")

        if self.state == State.SLEEPING:
            self.wakeup()
        if msg.level < self.level:  # FIXME : elsif ?
            # Connect from a lower-level fragment
            self.logger.debug(f"Connect from lower level fragment {msg} on {self.name}")
            self.neighbors_labels[msg.sender] = EdgeLabel.BRANCH
            self.post_msg(
                msg.sender,
                InitiateMessage(
                    self.name, self.level, self.fragment_identity, self.state
                ),
            )
            if self.state == State.FIND:
                self.find_count += 1
        else:
            if self.neighbors_labels[msg.sender] == EdgeLabel.BASIC:
                # Postpone message until our level is higher
                self.logger.debug(f"Postponing on {self.name} msg {msg}")
                self.post_msg(self.name, msg)
            else:
                new_id = self.neighbors_weights[msg.sender]  # FIXME: weight as id
                self.logger.debug(
                    f"New fragment creation on {self.name}  "
                    f" new id: {new_id} level: {self.level+1}"
                )
                # New
                self.post_msg(
                    msg.sender,
                    InitiateMessage(self.name, self.level + 1, new_id, State.FIND),
                )

    @register("initiate")
    def on_initiate(self, sender: str, msg: InitiateMessage, t: float):
        """
        An `initate` message starts the search for the minimum-weight outer edge for a
        fragment.

        This search is performed by:
        * propagating `initiate` messages on all `BRANCH` edges of the fragment
        * send test messages sequentially on `BASIC` edges,
        
        Parameters
        ----------
        sender
        msg
        t

        """
        self.logger.debug(f"Initiate from {msg.sender} on {self.name} : {msg}")
        self.level = msg.level
        self.fragment_identity = msg.id
        self.state = msg.state
        self.in_branch = msg.sender
        self.best_edge = None
        self.best_weight = float("inf")  # if self.mode == "min" else -float("inf")

        # Send initiate to all other neighbors
        for neighbor in self.neighbors_weights:
            if (
                neighbor == msg.sender
                or self.neighbors_labels[neighbor] != EdgeLabel.BRANCH
            ):
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
            if self.state == State.FIND:
                self.find_count += 1
        if self.state == State.FIND:
            self.test()

    def test(self):
        self.logger.debug(f"Test `BASIC` adjacent edges on {self.name}")
        basic_edges = [
            n for n, label in self.neighbors_labels.items() if label == EdgeLabel.BASIC
        ]
        if basic_edges:
            _, min_edge = min((self.neighbors_weights[n], n) for n in basic_edges)
            self.logger.debug(
                f"Found lowest weight basec edge {min_edge}, send test on id "
            )
            self.test_edge = min_edge
            self.post_msg(
                min_edge, TestMessage(self.name, self.level, self.fragment_identity)
            )
        else:
            # No `BASIC` adjacent edge here, can report directly
            self.logger.debug(
                f"No `BASIC` adjacent edge on {self.name}, can report directly"
            )
            self.test_edge = None
            self.report()

    @register("test")
    def on_test(self, sender, msg: TestMessage, t: float):
        """
        A `test` message is send over a `BASIC` edge, to check if that edge is an
        outgoing edge for the fragment.
        * If the node is not in the same fragment, it accept the test,
          meaning that it is indeed an outgoing edge
        * otherwise, either the fragment id is the same (and it's not an outgoing edge)
          or the local level is not high enough and the answer is postponed.

        Parameters
        ----------
        sender
        msg
        t

        """
        self.logger.debug(f"On test msg on {self.name} : {msg}")
        if self.state == State.SLEEPING:
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
                if self.neighbors_labels[msg.sender] == EdgeLabel.BASIC:
                    self.logger.debug(f"Label edge {msg.sender} - {self.name} as `REJECTED`")
                    self.neighbors_labels[msg.sender] = EdgeLabel.REJECTED
                if self.test_edge != msg.sender:
                    self.logger.debug(f"Reject test from {msg.sender} on {self.name}")
                    self.post_msg(msg.sender, RejectMessage(self.name))
                else:
                    self.logger.debug(f"Try test on next edge on {self.name}")

                    self.test()

    @register("reject")
    def on_reject(self, sender, msg, t: float):
        self.logger.debug(f"On reject msg from {msg.sender} on {self.name} : {msg}")
        if self.neighbors_labels[msg.sender] == EdgeLabel.BASIC:
            self.neighbors_labels[msg.sender] = EdgeLabel.REJECTED
        self.test()

    @register("accept")
    def on_accept(self, sender, msg, t: float):
        self.logger.debug(f"On accept msg from {msg.sender} on {self.name} : {msg}")
        self.test_edge = None
        if self.neighbors_weights[msg.sender] < self.best_weight:
            self.best_edge = msg.sender
            self.best_weight = self.neighbors_weights[msg.sender]
        self.report()

    def report(self):
        self.logger.debug(
            f"Report {self.name} find {self.find_count}  - {self.test_edge}"
        )
        if self.find_count == 0 and self.test_edge is None:
            self.state = State.FOUND
            self.post_msg(self.in_branch, ReportMessage(self.name, self.best_weight))

    @register("report")
    def on_report(self, sender, msg: ReportMessage, t: float):
        self.logger.debug(
            f"On report msg from {msg.sender} on {self.name} : {msg} {self.state}"
        )

        if msg.sender != self.in_branch:
            self.find_count -= 1
            if msg.weight < self.best_weight:  # FIXME: support min and max !
                self.logger.debug(
                    f"found better weight on {self.name}, keep reporting {msg}"
                )
                self.best_weight = msg.weight
                self.best_edge = msg.sender
                self.report()
        else:
            if self.state == State.FIND:
                # Postpone message:
                self.logger.debug(f"Postponing on {self.name} msg {msg}")
                self.post_msg(self.name, msg)
            else:
                if msg.weight > self.best_weight:
                    self.logger.debug(f"Changing root on {self.name} msg {msg}")
                    self.change_root()
                else:
                    if msg.weight == self.best_weight and msg.weight == float("inf"):
                        # handle termination
                        self.logger.info(f"Finished on  {self.name} msg {msg}")
                        self.state = State.DONE
                        self.stop()
                    else:
                        self.logger.debug(
                            f"Cannot finish on {self.name} {msg.weight} != {self.best_weight} "
                        )


    def change_root(self):
        if self.neighbors_labels[self.best_edge] == EdgeLabel.BRANCH:
            self.post_msg(self.best_edge, ChangeRootMessage(self.name))
        else:
            self.post_msg(self.best_edge, ConnectMessage(self.name, self.level))
            self.neighbors_labels[self.best_edge] = EdgeLabel.BRANCH

    @register("changeroot")
    def on_changeroot(self, sender, msg, t):
        self.logger.debug(
            f"On test changeroot msg from {msg.sender} on {self.name} : {msg}"
        )
        self.changeroot()