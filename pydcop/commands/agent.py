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
.. _pydcop_commands_agent:

pydcop agent
============

``pydcop agent`` runs one or several standalone agent(s).


Synopsis
--------

::

  pydcop agent --names <names>
               [--address <ip_address>] [--port <start_port>]
               --orchestrator <ip[:<port>]>
               [--uiport <start_uiport>]
               [--restart]
               [--delay <delay>]


Description
-----------

Starts one or several agents. No orchestrator is started, you must start it
separately  using the `:ref:`pydcop_commands_orchestrator` command.
The orchestrator might be started after or before the agents, if it is not
reachable when starting the agents, they will wait until it is available.

All agents are started in the same process and communicate with one another
using an embedded http server (each agent has its own http server). You can
run this command several time on different machines, all pointing to the same
orchestrator ; this allows to run large distributed systems.

The ui-server is a websocket server (one for each agent) that gives access to
an agent internal state. It is only needed if you intend to connect a graphical
user interface for an agent ; while it is very useful it also may have some
impact on performance and is better avoided when running a system with a large
number of agents.


See Also
--------

**Command:** :ref:`pydcop_commands_orchestrator`

**Tutorial:** :ref:`tutorials_deploying_on_machines`


Options
-------

``-n <names>`` / ``--names <names>``
  The names of the agent(s). Notice that this needs to match the name of the
  agents expected by the orchestrator.

``--orchestrator <ip[:<port>]>``
  The address of the orchestrator as <ip>:<port> where the port is optional
  and defaults to 9000.

``--address <ip_address>``
  Optional IP address the agent will listen on.
  If not given we try to use the primary IP address.

``-p <start_port>`` / ``--port <start_port>``
  The port on which the agent will listen for messages. If several agents
  names are started (when giving several names) this port is used for the
  first agent and increment for each subsequent agent. Defaults to 9001

``-i <start_uiport>`` / ``--uiport <start_uiport>``
  The port on which the ui-server will be listening (same behavior as
  ``--port`` when starting several agents). If not given, no ui-server will be
  started for these/this agent(s).

``--restart``
  When setting this flag, agent(s) will restarted when when they have all
  stopped. Useful when running `pydcop agent` as daemon on a remote machine.

``--delay <delay>``
  An optional delay between message delivery, in second. This delay
  only applies to algorithm's messages and is useful when you want to
  observe (for example with the GUI) the behavior of the algorithm at
  runtime.




Note that the agent's port defaults to 9001 to avoid conflicts with the
orchestrator, whose port defaults to 9000.

Examples
--------

Running a single agent on port 9001, with an ui-server on port 10001::

    pydcop -v 3 agent -n a1 --orchestrator 127.0.0.1:9000 --uiport 10001

Running 5 agents, listening on port 9011 - 9016 (without ui-server)::

    pydcop -v 3 agent -n a1 a2 a3 a4 a5 -p 9011 --orchestrator 127.0.0.1:9000


"""

import logging
from time import sleep
from typing import List

from pydcop.dcop.objects import AgentDef
from pydcop.infrastructure.orchestratedagents import OrchestratedAgent
from pydcop.infrastructure.communication import HttpCommunicationLayer

logger = logging.getLogger("pydcop.cli.agent")
force_stopped = False
agents = []


def set_parser(subparsers):

    parser = subparsers.add_parser("agent", help="Run one or several standalone agents")
    parser.set_defaults(func=run_cmd)
    parser.set_defaults(on_force_exit=on_force_exit)

    parser.add_argument(
        "-n",
        "--names",
        type=str,
        nargs="+",
        help="The name of the agent(s). This must match the "
        "name of the agents expected by the "
        "orchestrator.",
    )
    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help="IP address the orchestrator will listen on. If "
        "not given we try to use the primary IP address.",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=9001,
        help="The port on which the agent will listen for "
        "messages. If several agents are started "
        "(when giving several names) this port is used "
        "for the first agent and incremented for each "
        "following agent.",
    )
    parser.add_argument(
        "-o",
        "--orchestrator",
        type=str,
        help="The address of the orchestrator <ip>[:<port>]",
    )

    parser.add_argument(
        "-i",
        "--uiport",
        type=int,
        default=None,
        help="The port on which the ui-server will be listening"
        " (same behavior as``--port`` when starting "
        "several agents). If not given, no ui-server will "
        "be started for these/this agent(s).",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        default=False,
        help="When setting this flag, agent(s) will restarted"
        "when when they have all stopped. Useful when "
        "running `pydcop agent` as daemon on a remote "
        "machine.",
    )
    parser.add_argument(
        "--delay",
        default=None,
        type=float,
        help="an optional delay between message delivery, "
        " in second. This delay only applies to "
        "algorithm's messages and is useful when you "
        "want to observe (for example with the UI) the "
        "behavior of the algorithm at runtime",
    )
    parser.add_argument("--replication", default=False, action="store_true")
    parser.add_argument("--capacity", default=100, type=int)


def run_cmd(args):
    global agents
    if ":" in args.orchestrator:
        o_addr, o_port = args.orchestrator.split(":")
    else:
        o_addr, o_port = args.orchestrator, 9000

    names = list(args.names)
    if args.restart:
        while not force_stopped:
            agents = start_agents(
                names,
                o_addr,
                int(o_port),
                args.uiport,
                args.address,
                args.port,
                args.delay,
                args.replication,
                args.capacity,
            )

            # block until all agents have finished
            for agent in agents:
                agent.join()
            logger.info("All agents have stopped")
            if force_stopped:
                break
            logger.info("Wait before restarting")
            sleep(10)

    else:
        agents = start_agents(
            names,
            o_addr,
            int(o_port),
            args.uiport,
            args.address,
            args.port,
            args.delay,
            args.replication,
            args.capacity,
        )


def on_force_exit(_, __):
    print("FORCE EXIT")
    global agents, force_stopped
    force_stopped = True
    for agent in agents:
        agent.stop()


def start_agents(
    names: List[str],
    o_addr,
    o_port,
    u_port,
    a_addr,
    a_port,
    delay,
    replication,
    capacity,
):
    """
    Start orchestrated agents.

    Each agent will run in its own thread, in the same process. They are
    orchestrated by an orchestrator running in another process (which must be
    launched separately).

    Parameters
    ----------
    names: list of strings
        the names of the agents
    u_port: int
        start port for ui
    a_addr: str
        IP address the agent will listen on
    a_port: int
        start port for agents (messages)
    o_addr
        orchestrator address
    o_port
        orchestrator port

    Returns
    -------
    agents
        the list of orchestrated agents started

    """
    started_agents = []
    for a in names:
        if u_port:
            logger.info(
                "Starting agent {} on port {} with ui-server on {}".format(
                    a, a_port, u_port
                )
            )
        else:
            logger.info(
                "Starting agent {} on port {} without ui-server ".format(a, a_port)
            )

        comm = HttpCommunicationLayer((a_addr, a_port))
        agt_def = AgentDef(a, capacity=capacity)
        if replication:
            agent = OrchestratedAgent(
                agt_def,
                comm,
                (o_addr, o_port),
                ui_port=u_port,
                delay=delay,
                replication="dist_ucs_hostingcosts",
            )
        else:
            agent = OrchestratedAgent(
                agt_def, comm, (o_addr, o_port), ui_port=u_port, delay=delay
            )

        agent.start()
        started_agents.append(agent)
        a_port += 1
        if u_port:
            u_port += 1
    logger.info("All %s agents started", len(names))
    return started_agents
