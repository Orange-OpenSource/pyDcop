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
very simple event-bux mecanism.

"""
from collections import defaultdict
from functools import wraps
from typing import Callable


class EventDispatcher(object):
    """
    A very simple event dispatcher.

    """

    def __init__(self, enabled=False):
        self._cbs = defaultdict(lambda: [])
        self.enabled = enabled

    def send(self, topic, evt):
        if not self.enabled:
            return
        for cb in self._cbs[topic]:
            cb(topic, evt)
        all_cbs = list(self._cbs.items())
        eligibles = [
            cbs
            for s_topic, cbs in all_cbs
            if s_topic[-1] == "*" and topic.startswith(s_topic[:-1])
        ]
        for cbs in eligibles:
            for cb in cbs:
                cb(topic, evt)

    def subscribe(self, topic: str, cb: Callable):
        """
        Register a call back to topic.

        Parameters
        ----------
        topic: str
            a topic
        cb:
            a callback

        Returns
        -------
        callable:
            The registered callback.
        """
        self._cbs[topic].append(cb)
        return cb

    def unsubscribe(self, cb: Callable, topic: str = None):

        if topic is None:
            all_cbs = list(self._cbs.items())
            for s_topic, s_cbs in all_cbs:
                if cb in s_cbs:
                    s_cbs.remove(cb)
        # else:

    def reset(self):
        self._cbs.clear()


event_bus = EventDispatcher()

buses = defaultdict(lambda: EventDispatcher())


def get_bus(name: str):
    return buses[name]
