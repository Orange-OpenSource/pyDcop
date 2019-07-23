# BSD-3-Clause License
#
# Copyright 2018 Orange
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
.. _pydcop_commands_generate_scenario:


pydcop generate agents
======================

Generate a scenario.


Synopsis
--------
::

    pydcop generate scenario
                      --evts_counts <evts_counts>
                      --actions_counts <actions_counts>
                      --delay <delay>
                      --dcop_files <dcop files>
                      [--initial_delay <delay>]
                      [--end_delay <delay>]

Description
-----------

Generate a scenario for dynamic DCOP.

Currently only generates agent removal events.


Options
-------
``--evts_counts <evts_counts>``
  Number of events in the scenario

``--actions_counts <actions_counts>``
  Number of action in each event

``--delay <delay>``
  delay between two events

``--dcop_files <dcop files>``
  the problem agents are created for. It can be given as one or several files, whose
  content will be appended before parsing.

``--initial_delay <delay>``
  delay before first event

``--end_delay <delay>``
  delay after last event

Examples
========

generate a scenario with 3 events, for a DCOP which is given as two files.::

  pydcop generate scenario --evts_count 3 \
     --delay 10 --actions_count 2
     --dcop_files coloring_random.yaml coloring_random_agts.yaml



"""
import logging
import random
from typing import List

from pydcop.dcop.scenario import DcopEvent, Scenario, EventAction
from pydcop.dcop.yamldcop import yaml_scenario, load_dcop_from_file

logger = logging.getLogger("pydcop.cli.generate")


def init_cli_parser(parent_parser):
    parser = parent_parser.add_parser(
        "scenario", help="Generate a scenario for dynamic DCOP"
    )
    parser.set_defaults(func=generate)

    parser.add_argument("--evts_count", type=int, help="Number of events")
    parser.add_argument(
        "--actions_count",
        type=int,
        required=True,
        help="Number of action for each event",
    )
    parser.add_argument(
        "--delay", type=int, required=True, help="delay between two events"
    )
    parser.add_argument(
        "--initial_delay", type=int, default=20, help="delay before the first event"
    )
    parser.add_argument(
        "--end_delay", type=int, default=20, help="delay after the last event"
    )

    parser.add_argument(
        "--dcop_files", type=str, nargs="+", required=False,  help="dcop file(s)"
    )
    parser.add_argument(
        "dcop_files_end", type=str, nargs="*", metavar="FILE",
        help="dcop file(s)", default=None,
    )


def generate(args):
    logger.info("loading dcop from {}".format(args.dcop_files))

    if args.dcop_files:
        dcop_files = args.dcop_files
    elif args.dcop_files_end:
        dcop_files = args.dcop_files_end

    dcop = load_dcop_from_file(dcop_files)
    agents = list(dcop.agents)

    scenario = generate_scenario(
        args.evts_count,
        args.actions_count,
        args.delay,
        args.initial_delay,
        args.end_delay,
        agents,
    )

    serialized = yaml_scenario(scenario)

    if args.output:
        output_file = args.output
        with open(output_file, encoding="utf-8", mode="w") as fo:
            fo.write(serialized)
    else:
        print(serialized)


def generate_scenario(
    evts_count, actions_count, delay, initial_delay, end_delay, agents
) -> Scenario:
    agents = set(agents)
    events: List[DcopEvent] = []

    events.append(generate_delay("init", initial_delay))

    for i in range(evts_count):
        removed_agents = random.sample(agents, actions_count)
        agents.difference_update(removed_agents)
        actions = [EventAction("remove_agent", agent=agent) for agent in removed_agents]

        events.append(DcopEvent(f"e{i}", actions=actions))
        if i != evts_count-1:
            events.append(generate_delay(f"d{i}", delay))

    events.append(generate_delay("end", end_delay))

    return Scenario(events)


def generate_delay(i, lenght):
    return DcopEvent(i, delay=lenght)
