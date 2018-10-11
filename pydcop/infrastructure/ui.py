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
import logging
from threading import Thread

from websocket_server.websocket_server import WebsocketServer

from pydcop.infrastructure.computations import MessagePassingComputation, \
    VariableComputation, DcopComputation
from pydcop.infrastructure.Events import event_bus


class UiServer(MessagePassingComputation):
    """
    The UiServer is a server that is used by a GUI, running in another
    process, to access the agent state for display purposes.

    It is implemented as a MessagePassingComputation.
    """

    def __init__(self, agent, port):
        super().__init__('_ui_' + agent.name)
        self._agent = agent
        self.port = port
        self.logger = logging.getLogger('pydcop.agent.ui.'+agent.name)
        self.server = WebsocketServer(self.port, host='0.0.0.0')
        self.server.set_fn_new_client(self._new_client)
        self.server.set_fn_client_left(self._client_left)
        self.server.set_fn_message_received(self._message_received)
        self.t = Thread(target=self.server.run_forever, name='ws-'+agent.name)
        self.t.setDaemon(True)

        # Multicast is currently buggy in WebsocketServer, until it is fixed,
        # we keep track of client manually (see. issue #56
        # https://github.com/Pithikos/python-websocket-server/issues/56 )
        self._clients = []

        event_bus.subscribe('computations.cycle.*', self._cb_cycle)
        event_bus.subscribe('computations.value.*', self._cb_value)
        event_bus.subscribe('computations.message_rcv.*', self._cb_msg_rcv)
        event_bus.subscribe('computations.message_snd.*', self._cb_msg_snd)
        event_bus.subscribe('agents.add_computation.*', self._cb_add_comp)
        event_bus.subscribe('agents.rem_computation.*', self._cb_rem_comp)

    def on_message(self, var_name, msg, t):
        pass

    def on_start(self):
        self.logger.debug('Starting ui server on %s', self.port)
        self.t.start()

    def on_stop(self):
        """
        Called when stopping the computation, shutdown the ws server

        """
        self.logger.debug('Stopping ui server on %s', self.port)

        # Closing the server does not close the client side websocket :
        # Add an application-level close message for this.
        self._send_to_all_clients(json.dumps({"cmd": "close"}))

        self.server.shutdown()
        self.server.server_close()

    def _new_client(self, client, server):
        self.logger.debug('new client %s on %s', client , server)
        self._clients.append(client)

    def _client_left(self, client, server):
        # Called for every client disconnecting
        self.logger.debug('client left %s on %s', client, server)
        self._clients.remove(client)

    # Called when a client sends a message
    def _message_received(self, client, server, message):
        self.logger.debug('msg from %s :  %s', client, message)

        cmd = None
        try:
            message = json.loads(message)
            cmd = message['cmd']
        except ValueError:
            self.logger.error('Could not parse message %s', message)
            return

        if cmd == 'test':
            server.send_message_to_all(json.dumps({'cmd': 'test',
                                                   'data': 'foo'}))
        elif cmd == 'agent':
            server.send_message(
                client,
                json.dumps({'cmd': 'agent',
                            'agent': self._agent_data(self._agent)}))

        elif cmd == 'computations':
            computations = []
            for c in self._agent.computations():
                computations.append(self._computation(c))

            server.send_message(
                client,
                json.dumps({'cmd': 'computations',
                            'computations': self._computations()}))

    def _agent_data(self, agent):

        agt = {
            'name' : agent.name,
            'extra': agent.agent_def.extra_attr(),
            'computations': self._computations(),
            'replicas': [],  # TODO !!
            'address': str(agent.address),
            'is_orchestrator': agent.name == 'orchestrator'
        }
        if agent.agent_def:
            agt.update(agent.agent_def.extra_attr())
        return agt

    def _computations(self):
        computations = []
        for c in self._agent.computations():
            computations.append(self._computation(c))
        return computations

    def _computation(self, computation):
        """Build a map repr of a computation."""
        c_value = None
        c_name = computation.name
        neighbors = []
        c_algo, c_type= None, None
        c_msg_count, c_msg_size, c_cycles, footprint = 0, 0, 0, 0

        if isinstance(computation, DcopComputation):
            # a dcop computation, but not for a variable, for now let's
            # assume it's a factor
            c_type = 'factor'
            c_value = ''
            neighbors = list(computation.neighbors)
            c_algo = {'name': computation.computation_def.algo.algo,
                      'params': computation.computation_def.algo.params
                      }
            c_msg_count, c_msg_size = \
                self._agent.agt_metrics.computation_msg_rcv(computation.name)
            c_cycles = computation.cycle_count
            footprint = computation.footprint()

            if isinstance(computation, VariableComputation):
                c_type = 'variable'
                c_value = computation.current_value
                c_name = computation.variable.name

        return {
            'id': computation.name,
            'name' : c_name,
            'type': c_type,
            'value': c_value,
            'neighbors': neighbors,
            'algo': c_algo,
            'msg_count': c_msg_count,
            'msg_size': c_msg_size,
            'cycles': c_cycles,
            'footprint' : footprint
        }

    def _cb_cycle(self, topic, cycle_event):
        computation, cycles = cycle_event
        if self.is_local_computation(computation):
            self.logger.debug('send cycle event %s ', cycle_event)
            self._send_to_all_clients(
                json.dumps({'evt': 'cycle',
                            'computation': computation,
                            'cycles': cycles
                            }))

    def _cb_value(self, topic, value_event):
        computation, value = value_event
        if self.is_local_computation(computation):
            self.logger.debug('send value event %s ', value_event)
            self._send_to_all_clients(
                json.dumps({'evt': 'value',
                            'computation': computation,
                            'value': value
                            }))

    def _cb_msg_rcv(self, topic: str, msg_event):
        computation, msg_size = msg_event
        if self.is_local_computation(computation):
            # self.logger.debug('send msg_rcv event %s ', msg_event)
            self._send_to_all_clients(
                json.dumps({'evt': 'msg_rcv',
                            'computation': computation,
                            'msg_size': msg_size
                            }))

    def _cb_msg_snd(self, topic, msg_event):
        computation, msg_size= msg_event
        if self.is_local_computation(computation):
            # self.logger.debug('send msg_snd event %s ', msg_event)
            self._send_to_all_clients(
                json.dumps({'evt': 'msg_snd',
                            'computation': computation,
                            'msg_size': msg_size
                            }))

    def _cb_add_comp(self, topic: str, comp_evt):
        agent, computation = comp_evt
        if agent == self._agent.name and \
                self.is_local_computation(computation.name):
            self.logger.debug('send add computation event %s ', comp_evt)
            self._send_to_all_clients(
                json.dumps({'evt': 'add_comp',
                            'computation': self._computation(computation),
                            }))

    def _cb_rem_comp(self, topic: str, comp_evt):
        self.logger.debug(f"remove com evt {comp_evt} on topic {topic} , agent {self._agent.name}")
        agent, computation = comp_evt
        if agent == self._agent.name:
            self.logger.debug('send remove computation event %s ', comp_evt)
            self._send_to_all_clients(
                json.dumps({'evt': 'rem_comp',
                            'computation': computation,
                            }))

    def is_local_computation(self, computation: str):
        comp_names = [c.name for c in self._agent.computations()]
        return computation in comp_names

    def _send_to_all_clients(self, msg):
        for client in self._clients:
            self.server.send_message(client, msg)
