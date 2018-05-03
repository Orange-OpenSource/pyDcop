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


import logging

from pydcop.dcop.objects import AgentDef
from pydcop.infrastructure.orchestratedagents import OrchestratedAgent
from pydcop.infrastructure.communication import HttpCommunicationLayer

"""
The 'agent' dcop cli command runs one or several standalone agent(s).


Notes
-----

By standalone, we means agents without an Orchestrator, which may be run 
separately (and generally before) using the `pydcop orchestrator` command.

Agents created with this command communicate with one another using an 
embedded http server (each agent has its own http server). 
 
The ui-server is a websocket server (one for each agent) that gives access to 
an agent internal state. It is only needed if you intend to connect a graphical 
user interface for an agent, while it is very usefull it also may have some 
impact on performance and is better avoided when running a ssytem with a large 
number of agents.


Examples
--------

Running a single agent on port 9000, with an ui-server on port 10001 

   pydcop -v 3 agent -n a1 -p 9001 --uiport 10001 --orchestrator 127.0.0.1:9000

Running 5 agents, listing on port 9001 - 9006 (without ui-port):

    pydcop -v 3 agent -n a1 a2 a3 a4 a5 -p 9001 --orchestrator 127.0.0.1:9000


  
"""


logger = logging.getLogger('pydcop.cli.agent')


def set_parser(subparsers):

    parser = subparsers.add_parser('agent',
                                   help='Run one or several standalone agents')
    parser.set_defaults(func=run_cmd)

    parser.add_argument('-n', '--names', type=str, nargs='+',
                        help='The name of the agent(s). This must match the '
                             'name of the agents expected by the '
                             'orchestrator.')
    parser.add_argument('-p', '--port', type=int,
                        help='The port on which the agent will '
                             'be listening for messages. If several name are '
                             'used, this port will incremented for next agents')
    parser.add_argument('-i', '--uiport', type=int, default=None,
                        help='The port on which the ui-server will '
                             'be listening for messages. If several name are '
                             'used, this port will incremented for next '
                             'agents. If not given, no ui-server will be '
                             'started for these/this agent(s)')

    parser.add_argument('-o', '--orchestrator', type=str,
                        help='The address of the orchestrator <ip>:port')


def run_cmd(args):

    o_addr, o_port = args.orchestrator.split(':')
    o_port = int(o_port)
    a_port = args.port
    u_port = args.uiport
    for a in args.names:
        if u_port :
            logger.info(
                'Starting agent {} on port {} with ui-server on {}'.format(
                    a, a_port, u_port))
        else:
            logger.info(
                'Starting agent {} on port {} without ui-server '.format(
                    a, a_port))

        comm = HttpCommunicationLayer(('127.0.0.1', a_port))
        agt_def = AgentDef(a)
        agent = OrchestratedAgent(agt_def, comm, (o_addr, o_port),
                                  ui_port=u_port)

        agent.start()
        a_port += 1
        if u_port:
            u_port += 1
