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
import json

from pydcop.replication.dist_ucs_hostingcosts import UCSReplicateMessage
from pydcop.replication.dist_ucs_hostingcosts import ReplicationTracker
from pydcop.replication.path_utils import Path, PathsTable
from pydcop.utils.simple_repr import simple_repr, from_repr


def test_tracker_add_computations():
    tracker = ReplicationTracker()
    tracker.add(["c1", "c2", "c3"])

    assert not tracker.is_empty()
    assert set(tracker.in_progress()) == {"c1", "c2", "c3"}


def test_tracker_add_computation_several_time_must_increase_count():
    tracker = ReplicationTracker()
    tracker.add(["c1", "c2", "c3"])
    tracker.add(["c1", "c3"])

    assert not tracker.is_empty()
    assert tracker.replicating["c1"] == 2
    assert tracker.replicating["c2"] == 1
    assert tracker.replicating["c3"] == 2


def test_tracker_remove_computation_must_decrease_count():
    tracker = ReplicationTracker()
    tracker.add(["c1", "c2", "c3"])
    tracker.add(["c1", "c2", "c3"])
    tracker.remove(["c1", "c3"])

    assert not tracker.is_empty()
    assert not tracker.is_done("c1")
    assert not tracker.is_done("c2")
    assert not tracker.is_done("c3")
    assert tracker.replicating["c1"] == 1
    assert tracker.replicating["c2"] == 2
    assert tracker.replicating["c3"] == 1


def test_tracker_when_reaching_0_computation_is_removed():
    tracker = ReplicationTracker()
    tracker.add(["c1", "c2", "c3"])
    tracker.remove(["c1", "c3"])

    assert tracker.is_done("c1")
    assert tracker.replicating["c2"] == 1
    assert tracker.is_done("c3")

    tracker.remove(["c2"])
    assert tracker.is_empty()


def test_serialization_message():
    p1 = ("1", "2", "2")
    table = [(3, p1)]

    msg = UCSReplicateMessage("msg_type", 0, 0, p1, table, ["1", "2"], None, 5, 2, [])

    r = simple_repr(msg)
    assert r
    # simply taking a simple_repr does not ensure that the json serialization is valid
    json_msg = json.dumps(r)
    assert json_msg


def test_unserialization_msg():
    p1 = ("1", "2", "2")
    table = [(3, p1)]

    msg = UCSReplicateMessage("msg_type", 0, 0, p1, table, ["1", "2"], None, 5, 2, [])

    r = simple_repr(msg)
    assert r
    # simply taking a simple_repr does not ensure that the json serialization is valid
    json_msg = json.dumps(r)
    loaded = json.loads(json_msg)

    obtained = from_repr(loaded)
    assert obtained

    assert obtained == msg
