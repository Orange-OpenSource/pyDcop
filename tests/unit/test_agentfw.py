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


import time
import types
import unittest
from unittest.case import skip
from unittest.mock import MagicMock

from pydcop.infrastructure.communication import InProcessCommunicationLayer, \
    Messaging
from pydcop import infrastructure



class AgentFwTest(unittest.TestCase):

    @skip
    def test_sendmsg_counts(self):

        comm = infrastructure.communication.InProcessCommunicationLayer()
        a1 = infrastructure.Agent('a1', comm)
        a2 = infrastructure.Agent('a2', comm)
        comm.register(a1.name, a1)
        comm.register(a2.name, a2)

        self.assertEqual(a2._num_received, 0)
        self.assertEqual(a1._num_sent, 0)

        a1.send_msg('a1', 'a2', 'pouet')

        self.assertEqual(a2._num_received, 1)
        self.assertEqual(a2._num_sent, 0)
        self.assertEqual(a1._num_sent, 1)
        self.assertEqual(a1._num_received, 0)

        received = a2.q.get_nowait()
        self.assertEqual(received, ('a1', 'a2', 'pouet'))


    @skip
    def test_sendmsg_two_neighbors(self):

        comm = infrastructure.communication.InProcessCommunicationLayer()
        a1 = infrastructure.Agent('a1', comm)
        a2 = infrastructure.Agent('a2', comm)
        a3 = infrastructure.Agent('a3', comm)
        comm.register(a1.name, a1)
        comm.register(a2.name, a2)
        comm.register(a3.name, a3)

        # use monkey patching on instance to set the _on_start method on a1
        # simply send a message to all neighbors
        def a1_start(self):
            print('starting a1')
            for n in ['a2', 'a3']:
                print('sending msg to ' + n)
                self.send_msg('a1', n, 'msg')

        a1._on_start = types.MethodType(a1_start, a1)

        # Monkey patching a2 & a3 to check for message arrival
        a2.received = False
        a3.received = False

        def handle_message(self, sender, dest, msg):
            print('Receiving message ' + msg)
            if msg == 'msg':
                self.received = True
        a2._handle_message = types.MethodType(handle_message, a2)
        a3._handle_message = types.MethodType(handle_message, a3)

        # Running the test
        a1.start()
        a2.start()
        a3.start()
        time.sleep(0.1)

        self.assertTrue(a2.received)
        self.assertTrue(a3.received)

        a1.stop()
        a2.stop()
        a3.stop()

    @skip
    def test_sendRevievedFrom(self):

        comm = InProcessCommunicationLayer()
        a1 = infrastructure.Agent('a1', comm)
        a2 = infrastructure.Agent('a2', comm)
        comm.register(a1.name, a1)
        comm.register(a2.name, a2)

        # use monkey patching on instance to set the _on_start method on a1
        # simply send a message to a2
        def a1_start(self):
            print('starting a1')
            print('sending msg to ')
            self.send_msg('a1', 'a2', 'msg')

        a1._on_start = types.MethodType(a1_start, a1)

        # Monkey patching a2 to check for message arrival
        a2.received_from = None

        def handle_message(self, sender, dest, msg):
            print('Receiving message from {} : {}'.format(sender, msg))
            self.received_from = sender
        a2._handle_message = types.MethodType(handle_message, a2)

        # Running the test
        a1.start()
        a2.start()
        time.sleep(0.1)

        self.assertEqual(a2.received_from, 'a1')

        a1.stop()
        a2.stop()
