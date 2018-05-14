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
from unittest.mock import MagicMock, call

import pytest

from pydcop.infrastructure.agents import Agent
from pydcop.infrastructure.communication import UnknownAgent, \
    InProcessCommunicationLayer
from pydcop.infrastructure.discovery import Discovery, Directory, \
    UnknownComputation, DiscoveryException


@pytest.fixture
def standalone_discovery():
    discovery = Discovery('test', 'address')
    return discovery


@pytest.fixture
def directory_discovery():
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

    for c in agt1.computations():
        agt1.remove_computation(c.name)
    for c in agt1.discovery.agent_computations(agt1.name):
        agt1.discovery.unregister_computation(c)

    for c in agt2.computations():
        agt2.remove_computation(c.name)
    for c in agt2.discovery.agent_computations(agt2.name):
        agt2.discovery.unregister_computation(c)
    wait_run()

    agt1.stop()
    agt2.stop()
    agt_dir.stop()


def test_register_agent_without_publish(standalone_discovery):
    standalone_discovery.register_agent('agt1', 'addr1', publish=False)

    assert 'agt1' in standalone_discovery.agents()
    assert standalone_discovery.agent_address('agt1') == 'addr1'


def test_register_agent_publish_with_no_directory(standalone_discovery):
    standalone_discovery.register_agent('agt1', 'addr1')

    assert 'agt1' in standalone_discovery.agents()
    assert standalone_discovery.agent_address('agt1') == 'addr1'


def test_unregister_agent_without_publish(standalone_discovery):
    standalone_discovery.register_agent('agt1', 'addr1', publish=False)
    standalone_discovery.unregister_agent('agt1', publish=False)

    assert 'agt1' not in standalone_discovery.agents()
    with pytest.raises(UnknownAgent):
        standalone_discovery.agent_address('agt1')


def test_raises_on_address_for_unknown_agent(standalone_discovery):
    standalone_discovery.register_agent('agt1', 'addr1', publish=False)

    with pytest.raises(UnknownAgent):
        standalone_discovery.agent_address('agt2')


def test_register_agent_publish_on_directory(directory_discovery):
    agt_dir, agt_dis, _ = directory_discovery

    agt_dis.discovery.register_agent('agt1', 'addr1')
    wait_run()

    # Registration must be effective on both agents
    assert agt_dir.discovery.agent_address('agt1') == 'addr1'
    assert agt_dis.discovery.agent_address('agt1') == 'addr1'


def test_unregister_agent_publish_on_directory(directory_discovery):
    agt_dir, agt_dis, _ = directory_discovery

    agt_dis.discovery.register_agent('agt_new', 'addr_new')
    wait_run()
    agt_dis.discovery.unregister_agent('agt_new', 'addr_new')
    wait_run()

    # Un-Registration must be effective on both agents
    with pytest.raises(UnknownAgent):
        agt_dir.discovery.agent_address('agt_new')
    with pytest.raises(UnknownAgent):
        agt_dis.discovery.agent_address('agt_new')


def test_unregister_agent_with_computation_fails(directory_discovery):
    # When un-registering an agent, all computations for this agent must be
    # unregistered first otherwise we get a DiscoveryException.
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_agent('agt_new', 'addr_new')
    agt1.discovery.register_computation('comp_new', 'agt_new')

    with pytest.raises(DiscoveryException):
        agt1.discovery.unregister_agent('agt_new')


def test_subscribe_agent_cb(directory_discovery):
    agt_dir, agt_dis, _ = directory_discovery

    agt_dir.discovery.register_agent('agt_new', 'addr_new')
    wait_run()
    cb = agt_dis.discovery.subscribe_agent('agt_new', MagicMock())
    wait_run()
    cb.assert_called_once_with('agent_added', 'agt_new', 'addr_new')


def test_subscribe_agent_cb_one_shot(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_agent('agt_new', 'addr_new')
    wait_run()
    cb = agt2.discovery.subscribe_agent('agt_new', MagicMock(), one_shot=True)
    agt1.discovery.unregister_agent('agt_new', 'addr_new')

    wait_run()
    # The cb msut be called only once even though there was two events for
    # agent_new
    cb.assert_called_once_with('agent_added', 'agt_new', 'addr_new')


def test_subscribe_agent_cb_several(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery
    agt1.discovery.register_agent('agt_new', 'addr_new')
    wait_run()

    cb = agt2.discovery.subscribe_agent('agt_new', MagicMock(), one_shot=False)
    wait_run()
    cb.assert_called_with('agent_added', 'agt_new', 'addr_new')

    agt1.discovery.unregister_agent('agt_new', 'addr_new')
    wait_run()
    # The cb must be called twice, one for each event for agent_new
    cb.assert_called_with('agent_removed', 'agt_new', None)


def test_subscribe_agent_cb_called_once(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery
    agt1.discovery.register_agent('agt_new', 'addr_new')
    wait_run()

    cb = agt2.discovery.subscribe_agent('agt_new', MagicMock())
    wait_run()
    agt2.discovery.subscribe_agent('agt_new')
    wait_run()
    cb.reset_mock()

    agt1.discovery.register_agent('agt_new', 'addr_new2')
    wait_run()

    cb.assert_called_once_with('agent_added', 'agt_new', 'addr_new2')


def test_subscribe_agent_on_second_agt(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_agent('agt_new', 'addr_new')
    wait_run()
    cb = agt2.discovery.subscribe_agent('agt_new', MagicMock())
    wait_run()
    wait_run()

    # Registration must be effective on both agents
    assert agt_dir.discovery.agent_address('agt_new') == 'addr_new'
    assert agt2.discovery.agent_address('agt_new') == 'addr_new'
    cb.assert_called_once_with('agent_added', 'agt_new', 'addr_new')


def test_unsubscribe_all_cb(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery
    agt1.discovery.register_agent('agt_new', 'addr_new')

    cb = agt2.discovery.subscribe_agent('agt_new', MagicMock())
    cb2 = agt2.discovery.subscribe_agent('agt_new', MagicMock())

    wait_run()
    cb.assert_called_with('agent_added', 'agt_new', 'addr_new')
    cb2.assert_called_with('agent_added', 'agt_new', 'addr_new')
    cb.reset_mock()
    cb2.reset_mock()

    removed = agt2.discovery.unsubscribe_agent('agt_new')
    assert removed == 2
    agt1.discovery.unregister_agent('agt_new', 'addr_new')
    wait_run()
    cb.assert_not_called()
    cb2.assert_not_called()


def test_unsubscribe_one_cb(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery
    agt1.discovery.register_agent('agt_new', 'addr_new')

    cb = agt2.discovery.subscribe_agent('agt_new', MagicMock())
    cb2 = agt2.discovery.subscribe_agent('agt_new', MagicMock())

    wait_run()
    cb.reset_mock()

    removed = agt2.discovery.unsubscribe_agent('agt_new', cb)
    assert removed == 1
    agt1.discovery.unregister_agent('agt_new', 'addr_new')
    wait_run()
    cb.assert_not_called()


def test_subscribe_all_agents(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery
    # Once subscribed for all agents, agt2 should be aware of any agent in the
    # system without registering for an agent explicitely by its name.
    agt2.discovery.subscribe_all_agents()

    agt1.discovery.register_agent('agt_new1', 'addr_new')
    agt1.discovery.register_agent('agt_new2', 'addr_new')
    wait_run()
    assert 'agt_new1' in agt2.discovery.agents()
    assert 'agt_new2' in agt2.discovery.agents()

    agt1.discovery.unregister_agent('agt_new2')
    wait_run()
    assert 'agt_new2' not in agt2.discovery.agents()


def test_subscribe_all_agents_after(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_agent('agt_new1', 'addr_new1')
    agt1.discovery.register_agent('agt_new2', 'addr_new2')
    wait_run()

    # When subscribing for all agents, agt2 should be aware agents even if
    # they were registered before its subscription.
    cb = agt2.discovery.subscribe_all_agents(MagicMock())
    wait_run()

    assert 'agt_new1' in agt2.discovery.agents()
    assert 'agt_new2' in agt2.discovery.agents()
    cb.assert_has_calls(
        [call('agent_added', 'agt_new1', 'addr_new1'),
         call('agent_added', 'agt_new2', 'addr_new2'),
         call('agent_added', 'agt1', agt1.address)],
        any_order=True)

    agt1.discovery.unregister_agent('agt_new2')
    wait_run()
    assert 'agt_new2' not in agt2.discovery.agents()


def test_subscribe_all_agents_with_cb(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery
    # One subscribed for all agents, agt2 should be aware of any agent in the
    #  system without registering for an agent explicitely by its name.
    cb = agt2.discovery.subscribe_all_agents(MagicMock())

    agt1.discovery.register_agent('agt_new1', 'addr_new1')
    agt1.discovery.register_agent('agt_new2', 'addr_new2')
    wait_run()
    assert 'agt_new1' in agt2.discovery.agents()
    assert 'agt_new2' in agt2.discovery.agents()

    cb.assert_has_calls(
        [call('agent_added', 'agt_new1', 'addr_new1'),
         call('agent_added', 'agt_new2', 'addr_new2')])

    agt1.discovery.unregister_agent('agt_new2')
    wait_run()
    assert 'agt_new2' not in agt2.discovery.agents()


def test_list_computations():
    discovery = Discovery('test', 'addr_test')
    discovery.register_agent('a1', 'addr1')
    discovery.register_computation('c1', 'a1')
    discovery.register_computation('c2', 'a1')

    assert set(discovery.computations()) == {'c1', 'c2'}


def test_list_computations_filter(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    assert not agt1.discovery.computations()

    assert set(agt1.discovery.computations(include_technical=True)) == \
           {'_directory', '_discovery_agt1'}


def test_computation_agent():
    discovery = Discovery('test', 'addr_test')
    discovery.register_agent('a1', 'addr1')
    discovery.register_computation('c1', 'a1')
    discovery.register_computation('c2', 'a1')

    assert discovery.computation_agent('c1') == 'a1'
    assert discovery.computation_agent('c2') == 'a1'

    with pytest.raises(UnknownComputation):
        discovery.computation_agent('c3')


def test_agent_computations(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    assert not agt1.discovery.agent_computations(agt1.name)
    assert set(agt1.discovery.agent_computations(agt1.name,
                                                 include_technical=True)) == \
           {'_discovery_agt1'}

    agt1.discovery.register_agent(agt1.name, 'addr1')
    agt1.discovery.register_computation('c1', agt1.name)
    agt1.discovery.register_computation('c2', agt1.name)

    assert set(agt1.discovery.agent_computations(agt1.name)) == {'c1', 'c2'}
    assert set(agt1.discovery.agent_computations(agt1.name,
                                                 include_technical=True)) ==\
           {'_discovery_agt1', 'c1', 'c2'}


def test_register_computation(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    with pytest.raises(UnknownAgent):
        agt1.discovery.register_computation('c1', 'agt_test')

    agt1.discovery.register_agent('agt_test', 'address')
    wait_run()
    agt1.discovery.register_computation('c1', 'agt_test')
    wait_run()

    assert agt1.discovery.computation_agent('c1') == 'agt_test'

    # agt2 is not subscribed to c1, it does not it
    with pytest.raises(UnknownComputation):
        agt2.discovery.computation_agent('c1')

    # But the discovery on the agent hosting the directory must know
    assert agt_dir.discovery.agent_address('agt_test')  == 'address'
    assert agt_dir.discovery.computation_agent('c1') == 'agt_test'


def test_register_own_computation(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_computation('c1')
    wait_run()

    assert agt1.discovery.computation_agent('c1') == agt1.name
    assert agt_dir.discovery.computation_agent('c1') == agt1.name


def test_register_computation_with_agent_address(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_computation('c1', 'agt_test', 'addr_test')
    wait_run()

    assert agt1.discovery.agent_address('agt_test') == 'addr_test'
    assert agt_dir.discovery.agent_address('agt_test') == 'addr_test'


def test_unregister_computation(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_computation('c1', 'agt_test', 'addr_test')
    wait_run()
    agt1.discovery.unregister_computation('c1')
    wait_run()

    # Un-Registration must be effective on both agents
    with pytest.raises(UnknownComputation):
        agt_dir.discovery.computation_agent('c1')
    with pytest.raises(UnknownComputation):
        agt1.discovery.computation_agent('c1')


def test_unregister_computation_with_agent(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_computation('c1', 'agt_test', 'addr_test')
    wait_run()

    # c1 is not registered on agt_test2, must raise
    with pytest.raises(ValueError):
        agt1.discovery.unregister_computation('c1', 'agt_test2')

    # c2 is not registered on agt_test, must not raise
    agt1.discovery.unregister_computation('c2', 'agt_test')

    agt1.discovery.unregister_computation('c1', 'agt_test')
    wait_run()

    # Un-Registration must be effective on both agents
    with pytest.raises(UnknownComputation):
        agt_dir.discovery.computation_agent('c1')
    with pytest.raises(UnknownComputation):
        agt1.discovery.computation_agent('c1')


def test_subscribe_computation_no_cb(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_computation('c1', 'agt1')
    agt2.discovery.subscribe_computation('c1')
    wait_run()

    assert agt2.discovery.computation_agent('c1') == 'agt1'


def test_subscribe_computation_one_cb(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_computation('c1')
    cb = agt2.discovery.subscribe_computation('c1', MagicMock())
    wait_run()

    cb.assert_called_once_with('computation_added', 'c1', agt1.name)

    assert agt2.discovery.computation_agent('c1') == 'agt1'


def test_unsubscribe_computation_cb(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_computation('c1')
    cb = agt2.discovery.subscribe_computation('c1', MagicMock())
    wait_run()
    cb.reset_mock()

    agt2.discovery.unsubscribe_computation('c1', cb)
    cb.assert_not_called()
    agt1.discovery.register_computation('c1', 'agt_new', 'addr_new')
    assert agt2.discovery.computation_agent('c1') == agt1.name


def test_unsubscribe_computation_no_cb(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_computation('c1')
    cb = agt2.discovery.subscribe_computation('c1', MagicMock())
    wait_run()
    cb.reset_mock()

    agt2.discovery.unsubscribe_computation('c1', cb)
    cb.assert_not_called()
    agt1.discovery.register_computation('c1', 'agt_new', 'addr_new')
    assert agt2.discovery.computation_agent('c1') == agt1.name


def test_register_replica_for_unknown_replication_should_raise(
        directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    with pytest.raises(UnknownComputation):
        agt1.discovery.register_replica('c1', agt1.name)


def test_register_replica_local(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_computation('c1')
    agt1.discovery.register_replica('c1', agt1.name)
    wait_run()

    # Both agents must but be able to know who is hosting the replicas for c1
    assert agt1.discovery.replica_agents('c1') == {agt1.name}


def test_replica_should_be_visible_for_subscribed_agents(
        directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_computation('c1')
    agt1.discovery.register_replica('c1', agt1.name)
    agt2.discovery.subscribe_computation('c1')
    agt2.discovery.subscribe_replica('c1')
    wait_run()

    assert agt2.discovery.computation_agent('c1') == agt1.name

    # Both agents must but be able to know who is hosting the replicas for c1
    assert agt2.discovery.replica_agents('c1') == {agt1.name}


def test_replica_is_not_visible_for_not_subscribed_agents(
        directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_computation('c1')
    agt1.discovery.register_replica('c1', agt1.name)
    wait_run()

    # c2 is not subscribed to c&, should not even see the computation
    with pytest.raises(UnknownComputation):
        agt2.discovery.replica_agents('c1')

    agt2.discovery.subscribe_computation('c1')
    wait_run()

    # c2 is subscribed to computation c1, but not its replicas: should not
    # see the replica
    assert agt2.discovery.replica_agents('c1') == set()


def test_replica_removal_must_be_sent(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_computation('c1')
    agt1.discovery.register_replica('c1', agt1.name)
    agt2.discovery.subscribe_computation('c1')
    agt2.discovery.subscribe_replica('c1')
    wait_run()

    assert agt2.discovery.replica_agents('c1') == {agt1.name}

    agt1.discovery.unregister_replica('c1', agt1.name)
    wait_run()

    assert agt2.discovery.replica_agents('c1') == set()


def test_replica_removal_callback(directory_discovery):
    # make sure that registered callback is called when registering or
    # removing a replica

    agt_dir, agt1, agt2 = directory_discovery
    rep_cb = MagicMock()

    agt1.discovery.register_computation('c1')
    agt2.discovery.subscribe_computation('c1')
    agt2.discovery.subscribe_replica('c1', rep_cb)

    wait_run()

    agt1.discovery.register_replica('c1', agt1.name)
    wait_run()

    rep_cb.assert_called_once_with('replica_added', 'c1', agt1.name)
    rep_cb.reset_mock()

    agt1.discovery.unregister_replica('c1', agt1.name)
    wait_run()

    rep_cb.assert_called_once_with('replica_removed', 'c1', agt1.name)


def test_replica_unsubscribe(directory_discovery):
    agt_dir, agt1, agt2 = directory_discovery

    agt1.discovery.register_computation('c1')
    agt1.discovery.register_replica('c1', agt1.name)
    agt2.discovery.subscribe_computation('c1')
    agt2.discovery.subscribe_replica('c1')
    wait_run()

    assert agt2.discovery.replica_agents('c1') == {agt1.name}

    agt2.discovery.unsubscribe_replica('c1')
    agt1.discovery.unregister_replica('c1', agt1.name)
    wait_run()

    assert agt2.discovery.replica_agents('c1') == set()


def wait_run():
    # Small wait, just to give some slack for other threads to run.
    sleep(0.1)
