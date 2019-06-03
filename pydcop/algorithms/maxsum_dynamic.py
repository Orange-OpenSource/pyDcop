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

from pydcop.infrastructure.computations import Message, register
from pydcop.algorithms.amaxsum import MaxSumFactorComputation, MaxSumVariableComputation
from pydcop.algorithms.maxsum import MaxSumMessage
from pydcop.dcop.relations import NeutralRelation


class DynamicFunctionFactorComputation(MaxSumFactorComputation):

    """
    This is a specialisation of the computation performed for factor in the
    MaxSum algorithm, for factor whose function can change during the
    resolution process.

     The conditions under which the function changes are not specified in
     this class ; it must be subclassed.

     The the function changes, the variables it depends on must NOT change.
     Otherwise, the structure of the factor graph would need to be
     modified , with is not supported by this computation object.

     Of course, this computation object can be used like the original Maxsum
     factor computation object with non-changing factors,
     if change_factor_function is never called it works the same.

     FIXME
     -----

     This class does not work since the refactoring of maxsum implementation

     Parameters
     ----------
     factor: a factor object
     name: string
        an optional string, if not given the name of the factor will be used
        as the name of the computation
    msg_sender: Callable[[str, str, Message, int]
        A callable that can be used to send messages
    logger : a logger
        optional


     """

    def __init__(self, comp_def=None):
        super().__init__(comp_def=comp_def)

    def change_factor_function(self, fn):
        """

        :param fn: the new function for this factor.
        :return:
        """
        # Make sure the new function has the same dimension as the
        # previous one.
        if len(self.factor.dimensions) != len(fn.dimensions):
            raise ValueError(
                "Dimensions must be the same when changing "
                "function in DynamicFunctionFactorComputation"
            )
        diff1 = [v for v in self.factor.dimensions if v not in fn.dimensions]
        diff2 = [v for v in fn.dimensions if v not in self.factor.dimensions]
        if diff1 or diff2:
            raise ValueError(
                "Dimensions must be the same when changing "
                "function in DynamicFunctionFactorComputation"
            )

        # Dimensions are ok, change factor computation object and emit cost
        # messages
        self.factor = fn
        # return self._init_msg() # FIXME

    def __str__(self):
        return "Maxsum dynamic function Factor computation for " + self.factor.name

    def __repr__(self):
        return "Maxsum dynamic function Factor computation for " + self.factor.name


class FactorWithReadOnlyVariableComputation(DynamicFunctionFactorComputation):
    """
    A FactorWithReadOnlyVariableComputation is a specialized
    DynamicFunctionFactorComputation for factor whose relation depends on
    read-only variables.
    These factor subscribe to read-only variable and are notified when the
    value of one of these variable changes. Then, the relation is sliced on
    these read-only variables (with their known current value) an the sliced
    relation is used for optimization.

    It is for example used for user rule relation, where the relation to be
    optimized depends on a condition based on the value of sensor variables.

    """

    def __init__(self, relation, read_only_variables, name=None, msg_sender=None):

        self._relation = relation
        self._read_only_variables = read_only_variables
        self._read_only_values = {}

        # make sure the list of read-only variable is valid
        writable_vars = relation.dimensions[:]
        for v in read_only_variables:
            if v not in relation.dimensions:
                raise ValueError(
                    "Read only {} variable must be in relation "
                    "scope {}".format(v.name, relation.dimensions)
                )
            writable_vars.remove(v)

        # We start with a neutral relation until we have all values from
        # the read-only variables the condition depends on:
        self._sliced_relation = NeutralRelation(writable_vars, name=self._relation.name)
        super().__init__(self._sliced_relation, name=name, msg_sender=msg_sender)

    def on_start(self):
        # when starting, subscribe to all sensor variable used in the
        # condition of the rule
        for v in self._read_only_variables:
            self._msg_sender.post_msg(self.name, v.name, Message("SUBSCRIBE", None))
        super().on_start()

    @register("VARIABLE_VALUE")
    def _on_new_var_value_msg(self, var_name, msg, t):
        msg_count, msg_size = 0, 0

        value = msg.content
        if var_name not in [v.name for v in self._read_only_variables]:
            self.logger.error("Unexpected value from %s - %s ", var_name, value)
        self.logger.debug("Received new value for %s - %s ", var_name, value)

        self._read_only_values[var_name] = value

        if len(self._read_only_variables) == len(self._read_only_values):

            new_sliced = self._relation.slice(self._read_only_values)

            if hash(new_sliced) != hash(self._sliced_relation):
                self.logger.info("Changing factor function %s ", self.name)
                msg_count, msg_size = self.change_factor_function(new_sliced)
                self._sliced_relation = new_sliced
                self._active = True
            else:
                self.logger.info("Equivalent relation, no change %s ", self.name)

        else:
            self.logger.info(
                "Still waiting for values to evaluate the " "rule ",
                self._read_only_values,
            )

        return {"num_msg_out": msg_count, "size_msg_out": msg_size}


class DynamicFactorComputation(MaxSumFactorComputation):
    """
    Factor Computation for dynamic Max-Sum.

     Factor using this computation can have their function changed at
     run-time, and the new function is not required to have the same
     dimension at the previous one.
     When the dimension changes, ADD and REMOVE variable are sent to the
     affected variable nodes.

    If the relation of this computation depends on external variables, the
    DynamicFactorVariableComputation automatically subscribes to these
    variables and use their value to slice them out of the relation. The
    relation that is subject to optimization only depends on Variable those
    values could be changed (hence not on external variable).

    It is also possible to change the relation directly with
    change_factor_function.

    """

    def __init__(self, relation, name=None, msg_sender=None):

        self._relation = relation
        self._current_relation = relation

        # Check if the factor depends on external variables
        self._external_variables = {}
        for v in relation.dimensions:
            if hasattr(v, "value"):
                self._external_variables[v.name] = v

        if self._external_variables:
            external_values = {
                v.name: v.value for v in self._external_variables.values()
            }
            self._current_relation = self._relation.slice(external_values)

        super().__init__(self._current_relation, name=name, msg_sender=msg_sender)

    def on_start(self):
        # subscribe to external variable
        for v in self._external_variables.values():
            self.subscribe(v)
        return super().on_start()

    def change_factor_function(self, fn):
        msg_count, msg_size = 0, 0

        var_removed = [v for v in self._factor.dimensions if v not in fn.dimensions]
        var_added = [v for v in fn.dimensions if v not in self._factor.dimensions]
        if not var_removed and not var_added:
            # Dimensions have not changed, simply change factor object and emit
            # cost messages
            self.logger.info("Function change with no change in " "factor's dimension")
            self._factor = fn
            msg_count, msg_size = self._init_msg()
        else:
            self.logger.info(
                "Function change with new variables %s and " "removed variables %s",
                var_added,
                var_removed,
            )
            self._factor = fn
            for v in var_removed:
                if v.name in self._costs:
                    del self._costs[v.name]
                if v.name in self._prev_messages:
                    del self._prev_messages[v.name]
            for v in var_added:
                self._costs[v.name] = {d: 0 for d in v.domain}
            self._valid_assignments_cache = None

            if var_removed:
                msg_count, msg_size = self._send_remove_var_msg(var_removed)

            if var_added:
                c, s = self._send_add_var_msg(var_added)
                msg_count += c
                msg_size += s
            # FIXME : send costs to other variables ?

        return msg_count, msg_size

    @register("VARIABLE_VALUE")
    def _on_new_var_value_msg(self, var_name, msg, t):
        msg_count, msg_size = 0, 0
        value = msg.content
        if var_name not in self._external_variables:
            self.logger.error("Unexpected value from %s - %s ", var_name, value)
        self.logger.debug("Received new value for %s - %s ", var_name, value)

        self._external_variables[var_name].value = value
        external_values = {v.name: v.value for v in self._external_variables.values()}
        new_sliced = self._relation.slice(external_values)

        if hash(new_sliced) != hash(self._current_relation):
            self.logger.info("Changing factor function %s ", self.name)
            msg_count, msg_size = self.change_factor_function(new_sliced)
            self._current_relation = new_sliced
            self._active = True

    def _send_add_var_msg(self, var_added):
        """
        Send an ADD message to all variabled newly added to the scope of the
        factor.

        :param var_added: a list of variable name
        :return: a pair (msg_count, msg_size ) with the number of messages
        sent and their total size
        """
        msg_count, msg_size = 0, 0
        msg_debug = {}
        for v in var_added:
            costs_v = self._costs_for_var(v)
            msg = MaxSumMessage("ADD", {"costs": costs_v})
            self._msg_sender.post_msg(self.name, v.name, msg)
            msg_debug[v.name] = costs_v
            msg_size += msg.size
            msg_count += 1

        debug = "ADD VAR MSG {} \n".format(self.name)
        for dest, msg in msg_debug.items():
            debug += "  * {} -> {} : {}\n".format(self.name, dest, msg)
        self.logger.info(debug + "\n")

        return msg_count, msg_size

    def _send_remove_var_msg(self, var_removed):
        """
        Send a REMOVE message to all variables newly removed from the scope
        of the factor.

        :param var_removed: a list of variable name
        :return: a pair (msg_count, msg_size ) with the number of messages
        sent and their total size
        """
        msg_count, msg_size = 0, 0

        for v in var_removed:
            msg = MaxSumMessage("REMOVE", {})
            self._msg_sender.post_msg(self.name, v.name, msg)
            msg_size += msg.size
            msg_count += 1
        debug = "REMOVE VAR INIT MSG {} \n".format(self.name)
        for dest in var_removed:
            debug += "  * {} -> {} \n".format(self.name, dest)
        self.logger.info(debug + "\n")

        return msg_count, msg_size

    def subscribe(self, variable):
        self._msg_sender.post_msg(self.name, variable.name, Message("SUBSCRIBE", None))

    def unsubscribe(self, variable):
        self._msg_sender.post_msg(self.name, variable.name, Message("SUBSCRIBE", None))

    def __str__(self):
        return "Maxsum dynamic Factor computation for " + self._factor.name

    def __repr__(self):
        return "Maxsum dynamic Factor computation for " + self._factor.name


class DynamicFactorVariableComputation(MaxSumVariableComputation):
    """
    Variable computation for dynamic Max-Sum.

    This computation must be used for any variable which depends on a factor
    whose dimensions can change: it supports adding and removing dependent
    factors with ADD and REMOVE messages.


    """

    def __init__(self, variable, factor_names, msg_sender=None):

        super().__init__(variable, factor_names=factor_names, msg_sender=msg_sender)

    @register("REMOVE")
    def _on_remove_msg(self, factor_name, msg, t):
        self.logger.debug(
            "Received REMOVE msg from %s on var %s", factor_name, self.name
        )

        # The removed factor should always be in the list of our factors but we
        # might have not received any costs from him yet.
        try:
            self._factors.remove(factor_name)
        except ValueError:
            msg = "CANNOT remove factor {} from variable {}, not in {}".format(
                factor_name, self.name, self._factors
            )
            self.logger.error(msg)
            raise ValueError(msg)

        if factor_name in self._costs:
            del self._costs[factor_name]
            # FIXME : is it necessary to forget all previously sent messages ?
            self._prev_messages.clear()

        # Select a new value.
        self._current_value, self._current_cost = self._select_value()
        self.logger.debug(
            "On Remove msg,  Variable %s select value %s with " "cost %s",
            self.name,
            self._current_value,
            self._current_cost,
        )

        # Do not send init cost, we may still have costs from other factors !
        msg_count, msg_size = self._compute_and_send_costs(self.factors)

    @register("ADD")
    def _on_add_msg(self, factor_name, msg, t):
        self.logger.debug("Received ADD msg from %s : %s ", factor_name, msg.content)
        self._factors.append(factor_name)
        return self._on_cost_msg(factor_name, msg)
