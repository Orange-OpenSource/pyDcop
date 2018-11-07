from unittest.mock import MagicMock

import pytest


@pytest.fixture
def event_bus():
    from pydcop.infrastructure.Events import event_bus

    event_bus.reset()
    event_bus.enabled = True
    return event_bus


def test_simple_send(event_bus):

    cb = MagicMock()

    event_bus.subscribe("evt-name", cb)

    event_bus.send("evt-name", 42)
    cb.assert_called_once_with("evt-name", 42)


def test_cb_not_called_when_not_subscribed(event_bus):

    cb = MagicMock()
    event_bus.send("another", 48)
    cb.assert_not_called()


def test_do_not_receive_evt_once_unsubscribed(event_bus):
    cb = MagicMock()

    event_bus.subscribe("evts", cb)

    event_bus.send("evts", 42)

    cb.assert_called_once_with("evts", 42)

    cb.reset_mock()
    event_bus.unsubscribe(cb)

    event_bus.send("evts", 43)
    cb.assert_not_called()


def test_unsubscribe_from_single_topic(event_bus):

    cb1 = event_bus.subscribe("evts", MagicMock())
    cb2 = event_bus.subscribe("evts", MagicMock())

    event_bus.send("evts", 42)

    cb1.assert_called_once_with("evts", 42)
    cb2.assert_called_once_with("evts", 42)

    cb1.reset_mock()
    cb2.reset_mock()
    event_bus.unsubscribe(cb2)

    event_bus.send("evts", 43)
    cb1.assert_called_once_with("evts", 43)
    cb2.assert_not_called()


def test_several_subscribers(event_bus):

    cb1 = event_bus.subscribe("evt-name", MagicMock())
    cb2 = event_bus.subscribe("evt-name", MagicMock())
    cb3 = event_bus.subscribe("evt-name", MagicMock())

    event_bus.send("evt-name", 42)

    cb1.assert_called_once_with("evt-name", 42)
    cb2.assert_called_once_with("evt-name", 42)
    cb3.assert_called_once_with("evt-name", 42)


def test_receive_evt_from_sub_topics(event_bus):

    cb1 = event_bus.subscribe("a.b.*", MagicMock())

    # Event on sub-topic : must be received
    event_bus.send("a.b.c", 42)
    cb1.assert_called_once_with("a.b.c", 42)

    # Event on another topic : must not be received
    cb1.reset_mock()
    event_bus.send("a.another.d.f", 42)
    cb1.assert_not_called()

    # Event on another topic : must not be received
    cb1.reset_mock()
    event_bus.send("a.bb.f", 42)
    cb1.assert_not_called()
