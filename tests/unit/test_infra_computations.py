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

from importlib import import_module
from time import sleep
from unittest.mock import MagicMock, ANY

import pytest

from pydcop.algorithms import AlgorithmDef, ComputationDef, load_algorithm_module
from pydcop.computations_graph.constraints_hypergraph import VariableComputationNode
from pydcop.dcop.objects import Variable, AgentDef
from pydcop.infrastructure.agents import Agent
from pydcop.infrastructure.communication import InProcessCommunicationLayer
from pydcop.infrastructure.computations import (
    Message,
    message_type,
    MessagePassingComputation,
    register,
)
from pydcop.infrastructure.orchestratedagents import OrchestratedAgent
from pydcop.infrastructure.orchestrator import RunAgentMessage, Orchestrator
from pydcop.utils.simple_repr import simple_repr
from pydcop.utils.simple_repr import from_repr


def test_message():
    msg = Message("msg_type", "foo")
    assert msg.type == "msg_type"
    assert msg.content == "foo"


def test_message_serialization():
    msg = Message("msg_type")
    r = simple_repr(msg)
    obtained = from_repr(r)
    assert msg == obtained


def test_message_factory():

    MyMessage = message_type("my_msg", ["foo", "bar"])
    msg = MyMessage(42, 21)
    assert msg.type == "my_msg"
    assert msg.foo == 42
    assert msg.bar == 21

    with pytest.raises(ValueError):
        MyMessage(1, 2, 3)


def test_message_factory_kwargs():

    MyMessage = message_type("my_msg", ["foo", "bar"])
    msg = MyMessage(bar=42, foo=21)
    assert msg.type == "my_msg"
    assert msg.foo == 21
    assert msg.bar == 42

    with pytest.raises(ValueError):
        msg = MyMessage(bar=42, pouet=21)


def test_message_factory_serialization():

    MyMessage = message_type("my_msg", ["foo", "bar"])
    msg = MyMessage(42, 21)
    r = simple_repr(msg)
    print(r)
    obtained = from_repr(r)
    assert msg == obtained


def test_setting_message_sender_on_computation():

    c = MessagePassingComputation("c")

    c.message_sender = MagicMock()

    msg = Message("type")
    c.post_msg("target", msg)

    c.message_sender.assert_called_with("c", "target", msg, None, None)


def test_setting_message_sender_only_works_once():

    c = MessagePassingComputation("c")

    c.message_sender = MagicMock()
    with pytest.raises(AttributeError):
        c.message_sender = MagicMock()


def test_periodic_action_on_computation():

    a = Agent("a", MagicMock())

    class TestComputation(MessagePassingComputation):
        def __init__(self):
            super().__init__("test")
            self.mock = MagicMock()

        def on_start(self):
            self.add_periodic_action(0.1, self.action)

        def action(self):
            self.mock()

    c = TestComputation()

    a.add_computation(c)
    a.start()
    a.run()
    sleep(0.25)
    a.stop()

    assert c.mock.call_count == 2


def test_remove_periodic_action_on_computation():
    a = Agent("a", MagicMock())

    class TestComputation(MessagePassingComputation):
        def __init__(self):
            super().__init__("test")
            self.mock = MagicMock()

        def on_start(self):
            self.handle = self.add_periodic_action(0.1, self.action)

        def action(self):
            self.mock()

        def test_remove(self):
            self.remove_periodic_action(self.handle)

    c = TestComputation()

    a.add_computation(c)
    a.start()
    a.run()
    sleep(0.25)

    assert c.mock.call_count == 2

    c.test_remove()
    c.mock.reset_mock()
    sleep(0.5)

    c.mock.assert_not_called()

    a.stop()


def test_oneshot_delayed_action_on_computation():
    # To implement a one-shot action, add a periodic action and remove it
    # the first time it is called:
    a = Agent("a", MagicMock())

    class TestComputation(MessagePassingComputation):
        def __init__(self):
            super().__init__("test")
            self.mock = MagicMock()

        def on_start(self):
            self.handle = self.add_periodic_action(0.1, self.action)

        def action(self):
            self.mock()
            self.remove_periodic_action(self.handle)

    c = TestComputation()
    a.add_computation(c)
    a.start()
    a.run()
    sleep(0.25)

    # the action is remove on firts call  there must be one single call
    assert c.mock.call_count == 1

    c.mock.reset_mock()
    sleep(0.2)

    c.mock.assert_not_called()

    a.stop()


def test_several_periodic_action_on_computation():

    a = Agent("a", MagicMock())

    class TestComputation(MessagePassingComputation):
        def __init__(self):
            super().__init__("test")
            self.mock1 = MagicMock()
            self.mock2 = MagicMock()

        def on_start(self):
            self.add_periodic_action(0.1, self.action1)
            self.add_periodic_action(0.2, self.action2)

        def action1(self):
            self.mock1()

        def action2(self):
            self.mock2()

    c = TestComputation()

    a.add_computation(c)
    a.start()
    a.run()
    sleep(0.25)
    a.stop()

    assert 1 <= c.mock1.call_count <= 2
    assert c.mock2.call_count == 1


def test_periodic_action_not_called_when_paused():

    a = Agent("a", MagicMock())

    class TestComputation(MessagePassingComputation):
        def __init__(self):
            super().__init__("test")
            self.mock = MagicMock()

        def on_start(self):
            self.add_periodic_action(0.1, self.action)

        def action(self):
            self.mock()

    c = TestComputation()

    a.add_computation(c)
    a.start()
    a.run()
    sleep(0.25)
    assert 1 <= c.mock.call_count <= 2
    c.mock.reset_mock()

    a.pause_computations("test")
    sleep(0.25)
    assert c.mock.call_count == 0

    a.stop()


def test_register_handler_decorator():
    class TestComputation(MessagePassingComputation):
        def __init__(self):
            super().__init__("test")
            self.mock = MagicMock()

        @register("test_type")
        def on_msg(self, sender: str, msg: Message, t: float):
            self.mock()

    c = TestComputation()

    assert "test_type" in c._decorated_handlers


def test_handler_decorator():
    class TestComputation(MessagePassingComputation):
        def __init__(self):
            super().__init__("test")
            self.mock = MagicMock()

        @register("test_type")
        def on_msg(self, sender: str, msg: Message, t: float):
            self.mock(sender, msg, t)

    # Computation must be started for messages to be handled
    c = TestComputation()
    c.start()

    msg = Message("test_type")
    c.on_message("foo", msg, 0)

    c.mock.assert_called_once_with("foo", msg, 0)


def test_handler_decorator_not_called_before_start():
    class TestComputation(MessagePassingComputation):
        def __init__(self):
            super().__init__("test")
            self.mock = MagicMock()

        @register("test_type")
        def on_msg(self, sender: str, msg: Message, t: float):
            self.mock(sender, msg, t)

    c = TestComputation()

    msg = Message("test_type")
    c.on_message("foo", msg, 0)

    # Computation is NOT started, handled must NOT be called
    c.mock.assert_not_called()


def test_memory_footprint():
    # use maxsum as is has a computation_memory function defined
    maxsum_module = load_algorithm_module("amaxsum")
    from pydcop.computations_graph.factor_graph import (
        VariableComputationNode as FGVariableComputationNode,
    )

    v1 = Variable("v1", [1, 2])
    comp_def = ComputationDef(
        FGVariableComputationNode(v1, []),
        AlgorithmDef.build_with_default_param("amaxsum"),
    )
    comp = maxsum_module.MaxSumVariableComputation(comp_def=comp_def)

    # The variable has no neighbors : footprint is 0
    assert comp.footprint() == 0


def test_memory_footprint_from_import_module():
    # use maxsum as is has a computation_memory function defined
    maxsum_module = import_module("pydcop.algorithms.amaxsum")
    from pydcop.computations_graph.factor_graph import (
        VariableComputationNode as FGVariableComputationNode,
    )

    v1 = Variable("v1", [1, 2])
    comp_def = ComputationDef(
        FGVariableComputationNode(v1, []),
        AlgorithmDef.build_with_default_param("amaxsum"),
    )
    comp = maxsum_module.MaxSumVariableComputation(comp_def=comp_def)

    # The variable has no neighbors : footprint is 0
    assert comp.footprint() == 0


def test_memory_footprint_from_classic_import():
    # use maxsum as is has a computation_memory function defined
    import pydcop.algorithms.amaxsum as maxsum_module
    from pydcop.computations_graph.factor_graph import (
        VariableComputationNode as FGVariableComputationNode,
    )

    v1 = Variable("v1", [1, 2])
    comp_def = ComputationDef(
        FGVariableComputationNode(v1, []),
        AlgorithmDef.build_with_default_param("amaxsum"),
    )
    comp = maxsum_module.MaxSumVariableComputation(comp_def=comp_def)

    # The variable has no neighbors : footprint is 0
    assert comp.footprint() == 0


def test_fallback_memory_footprint():
    # use dsatuto as is has no computation_memory function defined
    dsa_module = load_algorithm_module("dsatuto")

    v1 = Variable("v1", [1, 2])
    comp_def = ComputationDef(
        VariableComputationNode(v1, []),
        AlgorithmDef.build_with_default_param("dsatuto"),
    )
    comp = dsa_module.DsaTutoComputation(comp_def)

    assert comp.footprint() == 1


def test_fallback_memory_footprint_from_import_module():
    # use dsatuto as is has no computation_memory function defined
    dsa_module = import_module("pydcop.algorithms.dsatuto")

    v1 = Variable("v1", [1, 2])
    comp_def = ComputationDef(
        VariableComputationNode(v1, []),
        AlgorithmDef.build_with_default_param("dsatuto"),
    )
    comp = dsa_module.DsaTutoComputation(comp_def)

    assert comp.footprint() == 1


def test_fallback_memory_footprint_from_classic_import():
    # use dsatuto as is has no computation_memory function defined
    import pydcop.algorithms.dsatuto as dsa_module

    v1 = Variable("v1", [1, 2])
    comp_def = ComputationDef(
        VariableComputationNode(v1, []),
        AlgorithmDef.build_with_default_param("dsatuto"),
    )
    comp = dsa_module.DsaTutoComputation(comp_def)

    assert comp.footprint() == 1


def test_fix_delayed_message_on_start():
    """
    Test case for bug fix:

    c1 send a message to c2 during its `on_start` handler. If c2 is not started yet
    at that point, the message is stored and injected once c2 starts.

    Sometime, when c2 starts in a2, it does not receive the pending message from c1.
    This is timing dependent and most of the time, the bug does not happens, especially
    if the on_start handlers contain log statements.

    The problem is due to the fact that start, and hence, the on_start handler and
    pending messages injection, might be done on the main thread, while message reception
    (and thus storage of th pending messages) is performed on the agent's thread.

    Everything that happens in an agent MUST run on the agent thread, including startup !

    To start a computation (on a started agent), one may call the `agt.run()` method,
    from any thread, in which case the computation is also started from that thread,
    and not from the agent thread.

    Currently (10/2019), there are only two ways for avoiding this:
    * use an orcherstrated agent, and start computation by sending it messages.
      However this also means using an orchestrator, which is not convenient during test
      and when working on non-dcop algorithm (the orchestrator is heavily biased
      toward DCOPs)
    * start all computations when starting the agent,
      using `a.start(run_computations=True)`, which is the workaround tested here.
    """
    class TestComputation1(MessagePassingComputation):
        def __init__(self):
            super().__init__("c1")

        def on_start(self):
            # self.logger.info("Starting c1, send to c2")
            self.post_msg("c2", Message("foo"))

    class TestComputation2(MessagePassingComputation):
        def __init__(self):
            super().__init__("c2")
            self.received = False

        # def on_start(self):
        #     # self.logger.info("Starting c2")
        #     pass

        @register("foo")
        def on_foo(self, sender, msg, t):
            self.logger.info(f"Receiving message {msg} from {sender}")
            self.received = True

    c1 = TestComputation1()
    c2 = TestComputation2()

    a1 = Agent("a1", InProcessCommunicationLayer())
    a1.add_computation(c1)
    a2 = Agent("a2", InProcessCommunicationLayer())
    a2.add_computation(c2)

    a1.discovery.register_computation("c2", "a2", a2.address)
    a2.discovery.register_computation("c1", "a1", a1.address)

    agts = [a1, a2]
    for a in agts:
        a.start(run_computations=True)

    sleep(0.5)

    for a in agts:
        a.stop()
    for a in agts:
        a.join()


    # Check that c2 really received the message
    assert c2.received

