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


from time import sleep
from unittest.mock import MagicMock

import pytest

from pydcop.infrastructure.computations import MessagePassingComputation, \
    Message, message_type
from pydcop.infrastructure.communication import InProcessCommunicationLayer
from pydcop.infrastructure.agents import Agent, AgentException
from pydcop.infrastructure.discovery import Directory, UnknownComputation


@pytest.fixture
def agent():
    agt = Agent('agt1', InProcessCommunicationLayer())
    yield agt
    agt.stop()


@pytest.fixture
def agents():
    # Agent hosting the directory
    agt_dir = Agent('agt_dir', InProcessCommunicationLayer())
    directory = Directory(agt_dir.discovery)
    agt_dir.add_computation(directory.directory_computation)
    agt_dir.discovery.use_directory('agt_dir', agt_dir.address)
    agt_dir.start()
    agt_dir.run(directory.directory_computation.name)

    # standard agents
    agt1 = Agent('agt1', InProcessCommunicationLayer())
    agt1.discovery.use_directory('agt_dir', agt_dir.address)
    agt1.start()

    agt2 = Agent('agt2', InProcessCommunicationLayer())
    agt2.discovery.use_directory('agt_dir', agt_dir.address)
    agt2.start()

    yield agt_dir, agt1, agt2

    agt1.stop()
    agt2.stop()
    agt_dir.stop()


PingMessage = message_type('ping', ['count'])


class PingComputation(MessagePassingComputation):

    def __init__(self, name: str, target: str=None):
        super().__init__(name)
        self.target = target
        self.ping_count = 0
        self._msg_handlers = {
            'ping' : self._on_ping
        }

    def on_start(self):
        if self.target is not None:
            ping_msg = PingMessage(1)
            self.post_msg(self.target, ping_msg)

    def _on_ping(self, var_name, msg, t):
        sleep(0.1)
        self.ping_count = msg.num + 1
        ping_msg = PingMessage(self.ping_count)
        self.post_msg(self.name, var_name, ping_msg)


def wait_run():
    # Small wait, just to give some slack for other threads to run.
    sleep(0.1)


def test_create():
    comm = InProcessCommunicationLayer()
    agent = Agent('agt1', comm)

    assert agent.name == 'agt1'
    assert not agent.computations()
    assert not agent.is_running
    assert agent.communication == comm
    assert agent.address == comm.address


def test_start(agent):

    agent.start()

    assert agent.is_running
    assert not agent.computations()
    # The discovery computation is deployed when starting the agent
    assert set(c.name for c in agent.computations(include_technical=True))\
        == {'_discovery_agt1'}


def test_stop(agent):

    agent._on_stop = MagicMock()

    agent.start()
    assert agent.is_running
    agent.stop()

    assert agent.is_stopping

    # must wait a little for the thread to terminate
    wait_run()

    assert not agent.is_running
    agent._on_stop.assert_called_once_with()


def test_add_computation_before_start(agent):
    ping = PingComputation('agt1')
    agent.add_computation(ping)

    assert agent.computation(ping.name) == ping
    assert not agent.computation(ping.name).is_running

    with pytest.raises(UnknownComputation):
        agent.computation('foo')


def test_add_computation_after_start(agent):
    agent.start()

    ping = PingComputation('agt1')
    agent.add_computation(ping)

    assert agent.computation(ping.name) == ping
    assert not agent.computation(ping.name).is_running


def test_run_computation(agent):
    ping = PingComputation('agt1')
    ping.on_start = MagicMock()

    agent.add_computation(ping)

    with pytest.raises(AgentException):
        agent.run()
    assert not agent.computation(ping.name).is_running
    ping.on_start.assert_not_called()

    agent.start()
    agent.run()
    wait_run()

    assert agent.computation(ping.name).is_running
    ping.on_start.assert_called_once_with()
    ping.on_start.reset_mock()

    # When calling run twice, the computation must not be started twice
    agent.run()
    wait_run()
    ping.on_start.assert_not_called()


def test_run_computation_by_name(agent):
    ping1 = PingComputation('p1')
    ping2 = PingComputation('p2')
    ping1.on_start = MagicMock()

    agent.add_computation(ping1)
    agent.add_computation(ping2)

    agent.start()
    agent.run(ping1.name)
    wait_run()

    # only ping1 was started, ping2 must not be running
    assert agent.computation(ping1.name).is_running
    assert not agent.computation(ping2.name).is_running
    ping1.on_start.assert_called_once_with()

    # starting an already running computation is armless
    agent.run(ping1.name)

    with pytest.raises(UnknownComputation):
        agent.run('bar')


def test_remove_computation(agent):
    ping1 = PingComputation('p1')
    ping1.on_stop = MagicMock()

    agent.add_computation(ping1)
    assert ping1 in agent.computations()
    agent.remove_computation(ping1.name)
    assert ping1 not in agent.computations()
    # p1 was not running, it must not be stopped:
    ping1.on_stop.assert_not_called()


def test_remove_running_computation(agent):
    ping1 = PingComputation('p1')
    ping1.on_stop = MagicMock()

    agent.add_computation(ping1)
    assert ping1 in agent.computations()
    agent.start()
    agent.run()
    wait_run()

    agent.remove_computation(ping1.name)
    assert ping1 not in agent.computations()
    assert not ping1.is_running
    # p1 was running, it must be stopped:
    ping1.on_stop.assert_called_once_with()


def test_pause_computation(agent):
    ping1 = PingComputation('p1')
    # ping1.on_stop = MagicMock()

    agent.add_computation(ping1)
    assert not ping1.is_paused

    with pytest.raises(AgentException):
        agent.pause_computations(ping1.name)

    agent.start()
    wait_run()

    agent.pause_computations(ping1.name)
    assert ping1.is_paused

    agent.unpause_computations(ping1.name)
    assert not ping1.is_paused


def test_pause_several_computations(agent):
    ping1 = PingComputation('p1')
    ping2 = PingComputation('p2')
    # ping1.on_stop = MagicMock()

    agent.add_computation(ping1)
    agent.add_computation(ping2)

    agent.start()
    wait_run()

    agent.pause_computations(ping1.name)
    agent.pause_computations(ping2.name)
    assert ping1.is_paused
    assert ping2.is_paused

    agent.unpause_computations(None)
    assert not ping1.is_paused
    assert not ping2.is_paused


def test_periodic_action(agent):
    mock = MagicMock()

    def cb(*args):
        mock(args)

    agent.set_periodic_action(0.1, cb)
    agent.start()
    sleep(0.25)
    # Depending on the start instant, the cb might be called 2 or 3 times:
    assert 2 <=len(list(mock.mock_calls)) <= 3


def test_periodic_action_not_called(agent):
    mock = MagicMock()
    def cb(*args):
        mock(args)
    agent.set_periodic_action(1, cb)
    agent.start()
    sleep(0.25)
    # Depending on the start instant, the cb might be called 2 or 3 times:

    assert not list(mock.mock_calls)


def test_several_periodic_action(agent):
    mock1 = MagicMock()

    def cb1(*args):
        mock1(args)

    mock2 = MagicMock()

    def cb2(*args):
        mock2(args)

    agent.set_periodic_action(0.1, cb1)
    agent.set_periodic_action(0.1, cb2)
    agent.start()
    sleep(0.25)
    # Depending on the start instant, the cb might be called 2 or 3 times:
    assert 2 <=len(list(mock1.mock_calls)) <= 3
    assert 2 <=len(list(mock2.mock_calls)) <= 3


def test_several_actions_different_period(agent):
    mock1 = MagicMock()

    def cb1(*args):
        mock1(args)

    mock2 = MagicMock()

    def cb2(*args):
        mock2(args)

    agent.set_periodic_action(0.1, cb1)
    agent.set_periodic_action(0.2, cb2)
    agent.start()
    sleep(0.5)
    # Depending on the start instant, the cb might be called 2 or 3 times:
    assert 4 <=len(list(mock1.mock_calls)) <= 5
    assert 2 <=len(list(mock2.mock_calls)) <= 3


def test_remove_action(agent):
    mock1 = MagicMock()

    def cb1(*args):
        mock1(args)

    handle = agent.set_periodic_action(0.1, cb1)
    agent.start()
    sleep(0.5)
    # Depending on the start instant, the cb might be called 2 or 3 times:
    assert 4 <=len(list(mock1.mock_calls)) <= 5

    agent.remove_periodic_action(handle)
    mock1.reset_mock()
    sleep(0.5)
    mock1.assert_not_called()
