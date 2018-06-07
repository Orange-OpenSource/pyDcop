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


import unittest
from http.server import HTTPServer
from threading import Thread
from time import sleep
from unittest.mock import MagicMock, create_autospec, call, ANY

import pytest
import requests

from pydcop.infrastructure.communication import Messaging, \
    InProcessCommunicationLayer, \
    MPCHttpHandler, HttpCommunicationLayer, ComputationMessage, \
    UnreachableAgent, MSG_MGT, UnknownAgent, UnknownComputation, MSG_ALGO
from pydcop.infrastructure.computations import Message
from pydcop.infrastructure.discovery import Discovery


def skip_http_tests():
    import os
    try:
        return os.environ['HTTP_TESTS'] == 'NO'
    except KeyError:
        return False


@pytest.fixture
def local_messaging():
    comm = InProcessCommunicationLayer()
    comm.discovery = Discovery('a1', 'addr1')
    messaging = Messaging('a1', comm)
    return messaging

class TestMessaging(object):

    def test_messaging_local_msg(self, local_messaging):

        local_messaging.discovery.register_computation('c1', 'a1')
        local_messaging.discovery.register_computation('c2', 'a1')

        msg = MagicMock()

        local_messaging.post_msg('c1', 'c2', msg)

        (src, dest, o_msg, type), t = local_messaging.next_msg()
        assert o_msg == msg
        assert dest, 'c2'
        assert src, 'c1'

    def test_retry_when_posting_msg_to_unknown_computation(
            self, local_messaging):
        local_messaging.discovery.register_computation('c1', 'a1')

        local_messaging.post_msg('c1', 'c2', 'a msg')

        # c2 is unknown, the message should not be in the queue
        full_msg, _ = local_messaging.next_msg()
        assert full_msg is  None

        # Register c2 : the message will now be delivered to the queue
        local_messaging.discovery.register_computation('c2', 'a1')
        (src, dest, full_msg, type), _ = local_messaging.next_msg()
        assert full_msg is 'a msg'

    def test_raise_when_posting_msg_from_unknown_computation(
            self, local_messaging):
        local_messaging.discovery.register_computation('c1', 'a1')
        local_messaging.discovery.register_computation('c2', 'a2', 'addr2')

        # Attempt to send a message to c2, from c3 which is not hosted locally
        with pytest.raises(UnknownComputation):
            local_messaging.post_msg('c3', 'c2', 'a msg')

    def test_next_message_returns_None_when_no_msg(self, local_messaging):
        local_messaging.discovery.register_computation('c1', 'a1')

        full_msg, _ = local_messaging.next_msg()
        assert full_msg is None

    def test_msg_to_computation_hosted_on_another_agent(self, local_messaging):

        local_messaging.discovery.register_computation('c1', 'a1')
        local_messaging.discovery.register_computation('c2', 'a2', 'addr2')
        local_messaging._comm.send_msg = MagicMock()

        msg = MagicMock()
        local_messaging.post_msg('c1', 'c2', msg)

        # Check that the msg was passed to the communication layer
        local_messaging._comm.send_msg.assert_called_with(
            'a1', 'a2',
            ComputationMessage('c1', 'c2', msg, ANY),
            on_error=ANY)

        # Check it's not in the local queue
        full_msg, _ = local_messaging.next_msg()
        assert full_msg is  None

    def test__metrics_local_msg(self, local_messaging):
        local_messaging.discovery.register_computation('c1', 'a1')
        local_messaging.discovery.register_computation('c2', 'a1')
        local_messaging.discovery.register_computation('c3', 'a1')

        msg = MagicMock()
        msg.size = 42

        local_messaging.post_msg('c1', 'c2', msg)

        assert local_messaging.count_all_ext_msg == 0
        assert local_messaging.size_all_ext_msg == 0

        msg2 = MagicMock()
        msg2.size = 12

        local_messaging.post_msg('c1', 'c3', msg2)

        assert local_messaging.count_all_ext_msg == 0
        assert local_messaging.size_all_ext_msg == 0

    def test__metrics_ext_msg(self, local_messaging):

        local_messaging.discovery.register_computation('c1', 'a1')
        local_messaging.discovery.register_computation('c2', 'a2', 'addr2')
        local_messaging.discovery.register_computation('c3', 'a1')
        local_messaging._comm.send_msg = MagicMock()

        msg = MagicMock()
        msg.size = 42

        local_messaging.post_msg('c1', 'c2', msg)

        assert local_messaging.size_ext_msg['c1'] == 42
        assert local_messaging.count_ext_msg['c1'] == 1
        assert local_messaging.count_all_ext_msg == 1
        assert local_messaging.size_all_ext_msg == 42

        msg2, msg3 = MagicMock(), MagicMock()
        msg2.size, msg3.size = 12, 5

        local_messaging.post_msg('c1', 'c2', msg2)
        local_messaging.post_msg('c1', 'c3', msg3)

        assert local_messaging.size_ext_msg['c1'] == 12 + 42
        assert local_messaging.count_ext_msg['c1'] == 2
        assert local_messaging.count_all_ext_msg == 2
        assert local_messaging.size_all_ext_msg == 42 + 12

    def test_do_not_count_mgt_messages(self, local_messaging):
        local_messaging.discovery.register_computation('c1', 'a1')
        local_messaging.discovery.register_computation('c2', 'a1')
        local_messaging._comm.send_msg = MagicMock()

        msg = MagicMock()
        msg.size = 42

        local_messaging.post_msg('c1', 'c2', msg, msg_type=MSG_MGT)

        assert local_messaging.count_all_ext_msg == 0
        assert local_messaging.size_all_ext_msg == 0


class TestInProcessCommunictionLayer(object):

    def test_address(self):
        # for in-process, the address is the object it-self
        comm1 = InProcessCommunicationLayer()
        assert comm1.address == comm1

    def test_addresses_are_not_shared_accross_instances(self):
        comm1 = InProcessCommunicationLayer()
        comm1.discovery = Discovery('a1', 'addr1')

        comm2 = InProcessCommunicationLayer()
        comm2.discovery = Discovery('a2', 'addr2')

        comm1.discovery.register_agent('a1', comm1)

        with pytest.raises(UnknownAgent):
            comm2.discovery.agent_address('a1')

    def test_msg_to_another_agent(self):

        comm1 = InProcessCommunicationLayer()
        Messaging('a1', comm1)
        comm1.discovery = Discovery('a1', comm1)

        comm2 = InProcessCommunicationLayer()
        Messaging('a2', comm2)
        comm2.discovery = Discovery('a2', comm2)
        comm2.receive_msg = MagicMock()

        comm1.discovery.register_agent('a2', comm2)

        full_msg = ('c1', 'c2', 'msg')
        comm1.send_msg('a1', 'a2', full_msg)

        comm2.receive_msg.assert_called_with('a1', 'a2', full_msg)

    def test_received_msg_is_delivered_to_messaging_queue(self):

        comm1 = InProcessCommunicationLayer()
        Messaging('a1', comm1)
        comm1.messaging.post_msg = MagicMock()

        comm1.receive_msg('a2', 'a1', ('c2', 'c1', 'msg', MSG_MGT))

        comm1.messaging.post_msg.assert_called_with('c2', 'c1', 'msg', 10)

    def test_raise_when_sending_to_unknown_agent_fail_default(self):
        comm1 = InProcessCommunicationLayer(on_error='fail')
        comm1.discovery = Discovery('a1', comm1)

        full_msg = ('c1', 'c2', 'msg', MSG_MGT)
        with pytest.raises(UnknownAgent):
            comm1.send_msg('a1', 'a2', full_msg)

    def test_raise_when_sending_to_unknown_agent_fail_on_send(self):
        comm1 = InProcessCommunicationLayer()
        comm1.discovery = Discovery('a1', comm1)

        full_msg = ('c1', 'c2', 'msg')
        with pytest.raises(UnknownAgent):
            comm1.send_msg('a1', 'a2', full_msg, on_error='fail')

    def test_ignore_when_sending_to_unknown_agent_ignore_default(self):
        comm1 = InProcessCommunicationLayer(on_error='ignore')
        comm1.discovery = Discovery('a1', comm1)

        full_msg = ('c1', 'c2', 'msg', MSG_MGT)
        assert comm1.send_msg('a1', 'a2', full_msg)

    def test_ignore_when_sending_to_unknown_agent_ignore_on_send(self):
        comm1 = InProcessCommunicationLayer()
        comm1.discovery = Discovery('a1', comm1)

        full_msg = ('c1', 'c2', 'msg')
        assert comm1.send_msg('a1', 'a2', full_msg,on_error='ignore')

    @pytest.mark.skip
    def test_retry_when_sending_to_unknown_agent_retry_default(self):
        comm1 = InProcessCommunicationLayer(on_error='retry')
        comm1.discovery = Discovery('a1', comm1)

        full_msg = ('c1', 'c2', 'msg')
        assert not comm1.send_msg('a1', 'a2', full_msg)

        comm2 = create_autospec(InProcessCommunicationLayer)
        comm1.discovery.register_agent('a2', comm2)

        comm2.receive_msg.assert_called_with('a1', 'a2', full_msg)
        comm2.receive_msg.assert_called_with('a1', 'a2', full_msg)

    @pytest.mark.skip
    def test_retry_when_sending_to_unknown_agent_retry_on_send(self):
        comm1 = InProcessCommunicationLayer(None)
        comm1.discovery = Discovery('a1', comm1)

        full_msg = ('c1', 'c2', 'msg')
        assert not comm1.send_msg('a1', 'a2', full_msg,on_error='retry')

        comm2 = create_autospec(InProcessCommunicationLayer)
        comm1.discovery.register_agent('a2', comm2)

        comm2.receive_msg.assert_called_with('a1', 'a2', full_msg)


@pytest.fixture
def httpd():
    server_address = ('127.0.0.1', 8001)
    httpd = HTTPServer(server_address, MPCHttpHandler)
    httpd.comm = MagicMock()
    yield httpd

    httpd.shutdown()
    httpd.server_close()


class TestHttpHandler(object):

    @pytest.mark.skipif(skip_http_tests(), reason='HTTP_TESTS == NO')
    def test_http_handler_one_message(self, httpd):

        t = Thread(name='http_thread',
                   target=httpd.serve_forever)
        t.start()
        requests.post('http://127.0.0.1:8001/test',
                      json={'key': 'value'},
                      timeout=0.5)

        sleep(0.5)

        httpd.comm.on_post_message.assert_called_once_with(
            '/test', None, None,
            ComputationMessage(
                src_comp=None,dest_comp=None,msg={'key': 'value'},
                msg_type=MSG_ALGO))

    @pytest.mark.skipif(skip_http_tests(), reason='HTTP_TESTS == NO')
    def test_http_handler_several_messages(self, httpd):

        t = Thread(name='http_thread',
                   target=httpd.serve_forever)
        t.start()
        requests.post('http://127.0.0.1:8001/test',
                      json={'key':'value'},
                      timeout=0.5)
        requests.post('http://127.0.0.1:8001/test2',
                      headers={'sender-agent': 'zero'},
                      json={'key':'value2'},
                      timeout=0.5)
        requests.post('http://127.0.0.1:8001/test3',
                      headers={'sender-agent': 'sender',
                               'dest-agent': 'dest',
                               'type': '15'},
                      json={'key':'value3'},
                      timeout=0.5)

        sleep(0.5)

        httpd.comm.on_post_message.assert_has_calls([
            call('/test', None, None,
                 ComputationMessage(src_comp=None,
                                    dest_comp=None,
                                    msg={'key': 'value'},
                                    msg_type=MSG_ALGO)),
            call('/test2', 'zero', None,
                 ComputationMessage(src_comp=None,
                                    dest_comp=None,
                                    msg={'key': 'value2'},
                                    msg_type=MSG_ALGO)),
            call('/test3', 'sender', 'dest',
                 ComputationMessage(src_comp=None,
                                    dest_comp=None,
                                    msg={'key': 'value3'},
                                    msg_type=15)),
            ])


@pytest.fixture
def http_comms():
    comm1 = HttpCommunicationLayer(('127.0.0.1', 10001))
    comm1.discovery = Discovery('a1', ('127.0.0.1', 10001))
    Messaging('a1', comm1)

    comm2 = HttpCommunicationLayer(('127.0.0.1', 10002))
    comm2.discovery = Discovery('a2', ('127.0.0.1', 10002))
    Messaging('a2', comm2)
    comm2.messaging.post_msg = MagicMock()

    yield  comm1, comm2
    comm1.shutdown()
    comm2.shutdown()


class TestHttpCommLayer(object):

    @pytest.mark.skipif(skip_http_tests(), reason='HTTP_TESTS == NO')
    def test_one_message_between_two(self, http_comms):
        comm1, comm2 = http_comms

        comm1.discovery.register_computation('c2', 'a2', ('127.0.0.1', 10002))
        comm2.discovery.register_computation('c1', 'a1', ('127.0.0.1', 10001))

        comm1.send_msg(
            'a1', 'a2',
            ComputationMessage('c1', 'c2', Message('test', 'test'), MSG_ALGO))

        comm2.messaging.post_msg.assert_called_with(
            'c1', 'c2', Message('test','test'), MSG_ALGO)

    @pytest.mark.skipif(skip_http_tests(), reason='HTTP_TESTS == NO')
    def test_several_messages_between_two(self, http_comms):
        comm1, comm2 = http_comms

        comm1.discovery.register_computation('c1', 'a2', ('127.0.0.1', 10002))
        comm2.discovery.register_computation('c2', 'a1', ('127.0.0.1', 10001))

        comm1.send_msg(
            'a1', 'a2',
            ComputationMessage('c1', 'c2', Message('test', 'test1'), MSG_ALGO))
        comm1.send_msg\
            ('a1', 'a2',
             ComputationMessage('c1', 'c2', Message('test', 'test2'), MSG_ALGO))
        comm1.send_msg(
            'a1', 'a2',
            ComputationMessage('c1', 'c2',Message('test','test3'), MSG_MGT))
        comm1.send_msg(
            'a1', 'a2',
            ComputationMessage('c1', 'c2',Message('test', 'test4'), MSG_ALGO))

        comm2.messaging.post_msg.assert_has_calls([
            call('c1', 'c2', Message('test', 'test1'), MSG_ALGO),
            call('c1', 'c2', Message('test', 'test2'), MSG_ALGO),
            call('c1', 'c2', Message('test', 'test3'), MSG_MGT),
            call('c1', 'c2', Message('test', 'test4'), MSG_ALGO),
            ])

    @pytest.mark.skipif(skip_http_tests(), reason='HTTP_TESTS == NO')
    def test_msg_to_unknown_computation_fail_mode(self, http_comms):
        comm1, comm2 = http_comms
        comm1.discovery.register_computation('c2', 'a2', ('127.0.0.1', 10002))

        comm2.discovery.register_computation('c1', 'a1', ('127.0.0.1', 10001))

        def raise_unknown(*args):
            raise UnknownComputation('test')
        comm2.messaging.post_msg = MagicMock(side_effect=raise_unknown)

        with pytest.raises(UnknownComputation):
            comm1.send_msg(
                'a1', 'a2',
                ComputationMessage('c1', 'c2', Message('a1', 't1'), MSG_ALGO),
                on_error='fail')

    @pytest.mark.skipif(skip_http_tests(), reason='HTTP_TESTS == NO')
    def test_msg_to_unknown_computation_ignore_mode(self, http_comms):
        comm1, comm2 = http_comms
        comm1.discovery.register_computation('c2', 'a2', ('127.0.0.1', 10002))

        comm2.discovery.register_computation('c1', 'a1', ('127.0.0.1', 10001))

        def raise_unknown(*args):
            raise UnknownComputation('test')
        comm2.messaging.post_msg = MagicMock(side_effect=raise_unknown)

        # Default mode is ignore : always returns True
        assert comm1.send_msg(
            'a1', 'a2',
            ComputationMessage('c1', 'c2', Message('a1', 'test1'), MSG_ALGO))

    @pytest.mark.skipif(skip_http_tests(), reason='HTTP_TESTS == NO')
    def test_msg_to_unknown_agent_fail_mode(self, http_comms):
        comm1, comm2 = http_comms
        # on a1, do NOT register a2, and still try to send a message to it
        with pytest.raises(UnknownAgent):
            comm1.send_msg(
                'a1', 'a2',
                ComputationMessage('c1', 'c2', Message('a1', 't1'), MSG_ALGO),
                on_error='fail')

    @pytest.mark.skipif(skip_http_tests(), reason='HTTP_TESTS == NO')
    def test_msg_to_unknown_agent_ignore_mode(self, http_comms):
        comm1, comm2 = http_comms

        # on a1, do NOT register a2, and still try to send a message to it
        # Default mode is ignore : always returns True
        assert comm1.send_msg(
            'a1', 'a2',
            ComputationMessage('c1', 'c2',Message('a1','t1'), MSG_ALGO))

    @pytest.mark.skipif(skip_http_tests(), reason='HTTP_TESTS == NO')
    def test_msg_to_unreachable_agent_fail_mode(self, http_comms):
        comm1, comm2 = http_comms
        # on a1, register a2 with the wrong port number
        comm1.discovery.register_computation('c2', 'a2', ('127.0.0.1', 10006))

        comm2.discovery.register_computation('c1', 'a1', ('127.0.0.1', 10001))

        with pytest.raises(UnreachableAgent):
            comm1.send_msg(
                'a1', 'a2',
                ComputationMessage('c1', 'c2', Message('a1', '1'), MSG_ALGO),
                on_error='fail')

    @pytest.mark.skipif(skip_http_tests(), reason='HTTP_TESTS == NO')
    def test_msg_to_unreachable_agent_ignore_mode(self, http_comms):
        comm1, comm2 = http_comms

        # on a1, register a2 with the wrong port number
        comm1.discovery.register_computation('c2', 'a2', ('127.0.0.1', 10006))
        comm2.discovery.register_computation('c1', 'a1', ('127.0.0.1', 10001))

        assert comm1.send_msg(
            'a1', 'a2',
            ComputationMessage('c1', 'c2', Message('a1', 't'), MSG_ALGO))
