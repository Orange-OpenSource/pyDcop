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


from pydcop.infrastructure.computations import Message, DcopComputation

"""
This module contains algorithm-agnostic computations, that is to say
computations that are not directly part of one specific DCOP algorithm but
may be used with several different DCOP algorithms.

"""


class ExternalVariableComputation(DcopComputation):

    def __init__(self, external_var, msg_sender=None, comp_def=None):
        super().__init__(external_var.name, comp_def)
        self._msg_handlers['SUBSCRIBE'] = self._on_subscribe_msg

        self._external_var = external_var.clone()
        self._msg_sender = msg_sender
        self.subscribers = set()

        self._external_var.subscribe(self._on_variable_change)

    @property
    def name(self):
        return self._external_var.name

    @property
    def current_value(self):
        return self._external_var.value

    def on_start(self):
        return {}

    def _on_variable_change(self, _):
        self._fire()

    def _on_subscribe_msg(self, var_name, _, t):
        self.subscribers.add(var_name)
        self._msg_sender.post_msg(self.name, var_name,
                                  Message('VARIABLE_VALUE',
                                          self._external_var.value))

    def change_value(self, value):
        self._external_var.value = value

    def _fire(self):
        for s in self.subscribers:
            self._msg_sender.post_msg(self.name, s,
                                      Message('VARIABLE_VALUE',
                                              self._external_var.value))

    def __str__(self):
        return 'External variable computation for ' + \
               self._external_var.name

    def __repr__(self):
        return 'External variable computation for ' + self._external_var.name
