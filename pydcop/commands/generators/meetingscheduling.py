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

.. _pydcop_commands_generate_meetings:

pydcop generate meetings
========================

Meetings scheduling benchmark problem generator
-----------------------------------------------

Synopsis
--------

::

  pydcop generate meetings
          --slots_count <slots_count>
          --events_count <events_count>
          --resources_count <resources_count>
          --max_resources_event <max_resources_event>
          [--max_length_event <max_length_event>]
          [--max_resource_value <max_resource_value>]


Description
-----------

This commands generate a meeting scheduling problem, based on
:cite:`maheswaran_taking_2004` with the *Private Event As Variable* (PEAV) model.

Note that this command generates both a DCOP and a distribution, as the PEAV model
also specifies the list of agents (one for each resource) and where each variable
is hosted.


**Note:** the generated DCOP and distribution are both written to the standard output.
To write in files, you can use the ``--output <file>``
:ref:`global option<usage_cli_ref_options>`.

Options
-------

``--slots_count <slots_count>``
  Total number of time slots

``--events_count``
  Number of events (aka meetings) to schedule

``--resources_count <resources_count>``
  Number of resources

``--max_resources_event <max_resources_event>``
  Maximum number of resources for each event: each event has a random
  number of requested resources in [1, max_resources_event]

``--max_length_event <max_length_event>``
  Maximum number of time slot for an event: each event has a random
  length in [1, max_length_event]. Optional, defaults to 1.

``--max_resource_value <max_resource_value>``
  Each resources has a random value in [1, max_resource_value] for
  each time slot and a value for being kept free (in [1, max_resource_value])
  at a given time slot. Optional, defaults to 10.


Examples
--------

Generating a meetings scheduling problem written directly to stdout::

    pydcop generate meetings --slots_count 5 \
        --events_count 4 --resources_count 3 --max_resources_event 2

Generating a meetings scheduling problem written in in ``meetings.yaml``. The
distribution is written in ``meetings_dist.yaml``::

    pydcop --output meetings.yaml generate meetings \\
        --slots_count 5 --events_count 6 --resources_count 3 \\
        --max_resources_event 2 --max_length_event 2

"""
import random
from os.path import splitext
from typing import Dict, List, Tuple, NamedTuple

import itertools

import yaml

from pydcop.dcop.dcop import DCOP
from pydcop.dcop.objects import Variable, Domain, AgentDef
from pydcop.dcop.relations import NAryMatrixRelation, Constraint
from pydcop.dcop.yamldcop import dcop_yaml
from pydcop.distribution.objects import Distribution


def init_cli_parser(parent_parser):
    parser = parent_parser.add_parser(
        "meetings", help="Generate a meeting scheduling benchmark problem"
    )
    parser.set_defaults(func=generate)

    parser.add_argument(
        "--slots_count", required=True, type=int, help="Total number of time slots"
    )
    parser.add_argument(
        "--events_count",
        required=True,
        type=int,
        help="Number of events (aka meetings) to schedule",
    )
    parser.add_argument(
        "--resources_count", required=True, type=int, help="Number of resources"
    )
    parser.add_argument(
        "--max_resources_event",
        required=True,
        type=int,
        help="Maximum number of resources for each event: each event has a random "
        "number of requested resources in [1, max_resources_event]",
    )
    parser.add_argument(
        "--max_length_event",
        required=False,
        default=1,
        type=int,
        help="Maximum number of time slot for an event: each event has a random "
        "length in [1, max_length_event]",
    )
    parser.add_argument(
        "--max_resource_value",
        required=False,
        default=10,
        type=int,
        help="Each resources has a random value in [1, max_resource_value] for "
        "each time slot and a value for being kept free "
        "(in [1, max_resource_value]) at a given time slot",
    )

    parser.add_argument(
        "--no_agents",
        default=False,
        required=False,
        action="store_true",
        help="generate the problem without any agents. You can use the 'pydcop generate " \
             "agents' to generate them with their hosting and route costs"
    )

    parser.add_argument(
        "--routes_default", type=int, required=False, help="Default routes cost"
    )

    parser.add_argument(
        "--hosting_default", type=int, required=False, help="Default hosting cost"
    )

    parser.add_argument(
        "--capacity", type=int, required=False, help="Capacity of agents"
    )

    # TODO: add support for 'Time Slot As Variable' and 'Events As Variables'
    # parser.add_argument(
    #     "--model",
    #     choices=["eav", "peav", "easv"],
    #     help="Model used for the meeting sheduling problem:"
    #     " 'eav' (Events As Variables),"
    #     " 'peav' (Private Events As Variables) or"
    # )

    # TODO: add support for intentional constraints
    # parser.add_argument(
    #     "--intentional",
    #     default=False,
    #     required=False,
    #     action="store_true",
    #     help="generate the problem in intentional form (default is extensive form)",
    # )


def generate(args):
    slots, events, resources = generate_problem_definition(
        args.slots_count,
        args.resources_count,
        args.max_resource_value,
        args.events_count,
        args.max_length_event,
        args.max_resources_event,
    )

    penalty = args.max_resource_value * args.slots_count * args.resources_count
    variables, constraints, agents = peav_model(slots, events, resources, penalty)

    domains = {variable.domain.name: variable.domain for variable in variables.values()}
    variables = {variable.name: variable for variable in variables.values()}
    # agents_defs = {agent.name: agent for agent, _ in agents.values()}
    # Generate agents hosting and route costs
    agents_defs = {}
    if not args.no_agents:
        for agent, agt_variables in agents.items():
            kw = {}
            kw["hosting_costs"] = {v.name: 0 for v in agt_variables}
            if args.hosting_default:
                kw["default_hosting_cost"] = args.hosting_default
            if args.capacity:
                kw["capacity"] = args.capacity
            if args.routes_default:
                kw["default_route"] = args.routes_default
            agents_defs[agent] = AgentDef(agent, **kw)

    dcop = DCOP(
        "MeetingSceduling",
        objective="max",
        domains=domains,
        variables=variables,
        constraints=constraints,
        agents=agents_defs,
    )

    if not args.no_agents:
        distribution = Distribution(
            {
                agent.name: [v.name for v in agents[agent.name]]
                for agent in agents_defs.values()
            }
        )

    if args.output:
        output_file = args.output
        with open(output_file, encoding="utf-8", mode="w") as fo:
            fo.write(dcop_yaml(dcop))

        if not args.no_agents:
            dist_result = {
                "inputs": {
                    "dist_algo": "peav",
                    "dcop": output_file,
                    "graph": "constraints_graph",
                    "algo": "NA",
                },
                "distribution": distribution.mapping(),
                "cost": None,
            }
            path, ext = splitext(output_file)
            dist_output_file = f"{path}_dist{ext}"
            with open(dist_output_file, encoding="utf-8", mode="w") as fo:
                fo.write(yaml.dump(dist_result))

    else:
        print(dcop_yaml(dcop))

        if not args.no_agents:
            dist_result = {
                "inputs": {
                    "dist_algo": "peav",
                    "dcop": "NA",
                    "graph": "constraints_graph",
                    "algo": "NA",
                },
                "distribution": distribution.mapping(),
                "cost": None,
            }
            # FIXME proper serialization of the distribution:
            print(yaml.dump(dist_result))


# Semantic type definitions:
EVT = int
RESOURCE = int
AGT = int
LENGTH = int
SLOT = int
VALUE = int


class Event(NamedTuple):
    id: EVT
    """Resources required for this event, with corresponding value"""
    resources: Dict[RESOURCE, VALUE]
    length: int


class Resource(NamedTuple):
    id: RESOURCE
    value_free: Dict[SLOT, VALUE]


def peav_model(
    slots: List[SLOT],
    events: Dict[EVT, Event],
    resources: Dict[RESOURCE, Resource],
    penalty,
) -> Tuple[
    Dict[Tuple[RESOURCE, EVT], Variable],
    Dict[str, Constraint],
    Dict[str, List[Variable]],
]:
    """
    In the PEAV model

    * agents represent resources


    Parameters
    ----------

    Returns
    -------

    """
    all_variables: Dict[Tuple[RESOURCE, EVT], Variable] = {}
    all_constraints: Dict[str, Constraint] = {}
    all_agents: Dict[str, List[Variable]] = {}

    # Each resource is represented by an agent, which controls one variable
    # for each event it could participate.
    for resource in resources.values():
        variables = peav_variables_for_resource(resource, events, len(slots))
        all_variables.update(variables)
        all_agents[f"a_{resource.id}"] = list(variables.values())

        constraints = peav_intra_extensive_constraints(
            resource, events, variables, penalty
        )
        all_constraints.update(constraints)

    # Generate inter-agent constraints: we have such constraint between any two
    # variables representing the same event for two different resources.
    for event in events.values():
        for resource_id1, resource_id2 in itertools.combinations(event.resources, 2):
            var1 = all_variables[(resource_id1, event.id)]
            var2 = all_variables[(resource_id2, event.id)]
            constraint = peav_inter_extensive_constraint(var1, var2, penalty)
            all_constraints[constraint.name] = constraint

    return all_variables, all_constraints, all_agents


def generate_problem_definition(
    slots_count: int,
    resources_count: int,
    max_resource_value: VALUE,
    events_count: int,
    max_length_event,
    max_resources_event,
) -> Tuple[List[SLOT], Dict[EVT, Event], Dict[RESOURCE, Resource]]:
    """
    Generate a  Multi-event scheduling problem definition.

    The definition is independent of the model used to map the problem to a DCOP.

    Parameters
    ----------
    slots_count
    resources_count
    max_resource_value
    events_count
    max_length_event
    max_resources_event

    Returns
    -------

    """
    slots = list(range(1, slots_count + 1))
    resources = generate_resources(resources_count, max_resource_value, slots)
    events = generate_events(
        events_count,
        max_resource_value,
        max_length_event,
        list(resources.values()),
        max_resources_event,
    )

    return slots, events, resources


def generate_resources(
    count: int, max_value: VALUE, slots: List[SLOT]
) -> Dict[RESOURCE, Resource]:
    resources: Dict[RESOURCE, Resource] = {}
    for i in range(count):
        # A resource has, for each time slot, a value if kept free:
        value_free = {j: random.randint(0, max_value) for j in slots}
        resources[i] = Resource(i, value_free)
    return resources


def generate_events(
    count: int,
    max_value: VALUE,
    max_length: int,
    resources: List[Resource],
    max_resources_count: int,
) -> Dict[EVT, Event]:
    events: Dict[EVT, Event] = {}
    for i in range(count):
        # Event's length:
        length = random.randint(1, max_length)
        # Resources required for this event:
        resources_count = random.randint(1, max_resources_count)
        event_resources = random.sample(resources, resources_count)
        # Value for each required resource for this event:
        values = {
            resource.id: random.randint(1, max_value) for resource in event_resources
        }
        events[i] = Event(i, values, length)
    return events


def peav_variables_for_resource(
    resource: Resource, events: Dict[EVT, Event], slots_count: int
) -> Dict[Tuple[RESOURCE, EVT], Variable]:
    variables: Dict[Tuple[RESOURCE, EVT], Variable] = {}
    for event in events.values():
        if resource.id in event.resources:
            name = f"v_{resource.id:02d}_{event.id:02d}"
            # The domain represents the start time (as slot) this event could start at.
            # Time slots start at 1, the value 0 represents a combination
            # (event, resource) that is not scheduled.
            domain = Domain(
                f"d_{name}",
                "time_slot",
                values=range(0, slots_count - event.length + 2),
            )
            variables[resource.id, event.id] = Variable(name, domain)
    return variables


def peav_intra_extensive_constraints(
    resource: Resource,
    events: Dict[EVT, Event],
    variables: Dict[Tuple[RESOURCE, EVT], Variable],
    penalty,
):
    resource_events_count = len(variables)
    constraints = {}
    for (resource_id1, event_id1), (resource_id2, event_id2) in itertools.combinations(
        variables, 2
    ):
        # As we are generating intra-agent constraint and agents map to resources in
        # the peav model, all resources must be the same
        assert resource.id == resource_id1 == resource_id2
        constraint = peav_intra_extensive_constraint(
            resource,
            events[event_id1],
            variables[(resource.id, event_id1)],
            events[event_id2],
            variables[resource.id, event_id2],
            penalty,
            resource_events_count,
        )
        constraints[constraint.name] = constraint

    if len(variables) == 1:
        # If there is a single variable (and thus a single event) for this resource,
        # we add a unary constraint which will account for the utility of scheduling
        # the resource on this single event. Otherwise, as these utilities are given by
        # internal binary variables, it would not be accounted for.
        # In Maheswaran_2012, this is done by introducing a dummy variable === 0,
        # to keep an artificial binary constraint. The result is the same but the
        #  unary-constraint approach makes more sense to me and fits pydcop better.
        (_, event_id), variable = variables.popitem()
        event = events[event_id]
        constraint = NAryMatrixRelation([variable], name=f"cu_{variable.name}")
        for t in variable.domain:
            value = resource_value_for_event(resource, event, t)
            constraint = constraint.set_value_for_assignment({variable.name: t}, value)
            constraints[constraint.name] = constraint

    return constraints


def peav_intra_extensive_constraint(
    resource: Resource,
    event1: Event,
    var1: Variable,
    event2: Event,
    var2: Variable,
    penalty: int,
    resource_events_count: int,
) -> Constraint:

    constraint = NAryMatrixRelation([var1, var2], name=f"ci_{var1.name}_{var2.name}")

    # For each possible partial assignment (t1, t2) to (var1, var2)
    # we compute the utility (or penalty)
    for t1 in var1.domain:
        for t2 in var2.domain:
            value = peav_intra_extensive_constraint_value(
                resource, event1, event2, penalty, resource_events_count, t1, t2
            )
            constraint = constraint.set_value_for_assignment(
                {var1.name: t1, var2.name: t2}, value
            )
    return constraint


def peav_intra_extensive_constraint_value(
    resource: Resource,
    event1: Event,
    event2: Event,
    penalty: int,
    resource_events_count: int,
    t1: SLOT,
    t2: SLOT,
) -> float:
    """
    Compute the value of an intra-agent constraint for assignment of a resource to two
    events scheduled at  (t1, t2).

    Parameters
    ----------
    resource: Resource
        the resource scheduled
    event1: Event
        The first scheduled event
    event2: Event
        The second scheduled event
    penalty: int
        Penalty in case of conflict
    resource_events_count
        The number of events this resources is participating to.
    t1: int
        schedule for first event
    t2: int
        schedule for second event

    Returns
    -------
    The value of the constraint for assignment (t1, t2)
    """

    if event1 == event2 and t1 != t2:
        # Penalty if two events are scheduled at different time by two
        # different resources/agents:
        return -penalty
    elif event1 != event2:
        # Intra-agent constraint: penalty if there is a schedule conflict.
        if t1 != 0 and t2 != 0 and t1 <= t2 <= t1 + event1.length - 1:
            return -penalty
        elif t1 != 0 and t2 != 0 and t2 <= t1 <= t2 + event2.length - 1:
            return -penalty
        else:
            # If there is no conflict: utility
            value = (
                1
                / (resource_events_count - 1)
                * (
                    resource_value_for_event(resource, event1, t1)
                    + resource_value_for_event(resource, event2, t2)
                )
            )
            return value
    else:
        raise Exception("Bug!")


def peav_inter_extensive_constraint(var1, var2, penalty):

    constraint = NAryMatrixRelation([var1, var2], name=f"ce_{var1.name}_{var2.name}")

    # For each possible partial assignment (t1, t2) to (var1, var2)
    # we compute the utility (or penalty)
    for t1 in var1.domain:
        for t2 in var2.domain:
            if t1 != t2:
                constraint = constraint.set_value_for_assignment(
                    {var1.name: t1, var2.name: t2}, -penalty
                )
    return constraint


def resource_value_for_event(resource: Resource, event: Event, t: SLOT) -> float:
    """
    The utility of affecting a resource to a given event.

    This utility is defined as the difference between the value of affecting the
    resource for all the time slots of the event and the aggregate
    value of the time slots if left free.

    Parameters
    ----------
    resource: Resource
        the resource
    event: Event
        the event
    t: int
        time slot

    Returns
    -------
    the utility of affecting the resource to this event at time slot t.
    """
    if t == 0:
        return 0
    evt_value = event.resources[resource.id] * event.length
    resource_value_if_free = sum(
        [resource.value_free[t + j] for j in range(0, event.length)]
    )
    return evt_value - resource_value_if_free
