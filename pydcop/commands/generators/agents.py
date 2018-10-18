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
.. _pydcop_commands_generate_agents:

pydcop generate agents
======================

generate a set of agents

"""
import logging

from pydcop.dcop.objects import AgentDef
from pydcop.dcop.yamldcop import yaml_agents

logger = logging.getLogger("pydcop.cli.generate")

COLORS = ["R", "G", "B", "O", "F", "Y", "L", "C"]


def init_cli_parser(parent_parser):
    parser = parent_parser.add_parser("agents", help="Generate a set of agents")
    parser.set_defaults(func=generate)

    parser.add_argument("--count", type=int, required=True, help="Number of agents")
    parser.add_argument(
        "--capacity", type=int, required=True, help="Capacity of agents"
    )


def generate(args):
    agents = generate_agents(args.count, args.capacity)
    serialized = yaml_agents(agents)

    if args.output:
        output_file = args.output
        with open(output_file, encoding="utf-8", mode="w") as fo:
            fo.write(serialized)
    else:
        print(serialized)


def generate_agents(agent_count: int, capacity: int):
    agents = []
    for i in range(agent_count):
        agt = AgentDef(f"a{i:02d}", capacity=capacity)
        agents.append(agt)
    return agents
