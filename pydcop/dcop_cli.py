#!/usr/bin/env python3

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
Main command-line interface for pydcop.


"""
import logging
import signal
import argparse
from os import path
import sys
from logging.config import fileConfig
from threading import Timer

import functools

from pydcop.version import __version__
from pydcop.commands import solve, orchestrator, agent, replica_dist, batch
from pydcop.commands import distribute
from pydcop.commands import graph
from pydcop.commands import generate
from pydcop.commands import run
from pydcop.commands import consolidate

cli_timer = None
TIMEOUT_SLACK = 40


def main():

    parser = argparse.ArgumentParser(description='pydcop')
    parser.add_argument('-v', '--verbose', default='0',
                        choices=[0, 1, 2, 3], type=int,
                        help='verbosity, 0 means only errors will be '
                             'logged, useful when you need to parse '
                             'the result automatically.')
    parser.add_argument('--version', action='store_true',
                        help='output pydcop version')
    parser.add_argument('-t', '--timeout', default=0, type=int,
                        help='timeout for running the command , '
                             'if not specified run until finished or stop '
                             'with ctrl-c.')
    parser.add_argument('--strict_timeout', default=0, type=int,
                        help='timeout for running the command , '
                             'if not specified run until finished or stop '
                             'with ctrl-c.')

    parser.add_argument('--output', type=str,
                        help='output file')
    parser.add_argument('--log', type=str,
                        help='log configuration file')

    subparsers = parser.add_subparsers(title='Actions', dest='action',
                                       description='To get help on a command, '
                                                   'use pydcop <command> -h')

    # Register commands for dcop cli
    solve.set_parser(subparsers)
    distribute.set_parser(subparsers)
    graph.set_parser(subparsers)
    agent.set_parser(subparsers)
    orchestrator.set_parser(subparsers)
    generate.set_parser(subparsers)
    replica_dist.set_parser(subparsers)
    run.set_parser(subparsers)
    batch.set_parser(subparsers)
    consolidate.set_parser(subparsers)

    # parse command line options
    args = parser.parse_args()

    if args.version:
        print('pydcop', __version__ )
        return

    _configure_logs(args.verbose, args.log)

    if hasattr(args, 'on_force_exit'):
        signal.signal(signal.SIGINT,
                      functools.partial(_on_force_exit, args.on_force_exit))

    if hasattr(args, 'func'):
        if args.timeout or args.strict_timeout:
            if hasattr(args, 'on_timeout'):
                global cli_timer

                if args.strict_timeout:
                    # print(f"Starting strict timeout timer {args.strict_timeout}")
                    cli_timer = Timer(args.strict_timeout, _on_timeout,
                                      [args.on_timeout])
                    timeout = args.strict_timeout
                elif args.timeout:
                    # the timeout slack is used to properly shutdown running threads
                    # print(f"Starting timeout timer {args.timeout}")
                    cli_timer = Timer(args.timeout + TIMEOUT_SLACK, _on_timeout,
                                      [args.on_timeout])
                    timeout = args.timeout

                cli_timer.daemon = True
                cli_timer.start()
                args.func(args, cli_timer, timeout)
            else:
                print('Command {}, does not support the global timeout '
                      'parameter'.format(args))
                sys.exit(2)
        else:
            args.func(args)
    else:
        print('Invalid command, for help use --help')
        parser.parse_args(['-h'])
        sys.exit(2)

    if not args:
        print('Invalid command, for help use --help')
        parser.parse_args(['-h'])
        sys.exit(2)


def _on_force_exit(sub_exit_func, sig, frame):

    if cli_timer is not None:
        cli_timer.cancel()

    sub_exit_func(sig, frame)


def _on_timeout(on_timeout_func):
    on_timeout_func()


def _configure_logs(level: int, log_conf: str):
    if log_conf is not None:
        if not path.exists(log_conf):
            raise ValueError(f"Could not find log configuration file {log_conf}")
        fileConfig(log_conf)
        logging.info(f'Using log config file {log_conf}')
        return

    # Default: remove all logs except error
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    # Format logs with hour and ms, but no date
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        '%H:%M:%S')
    console_handler.setFormatter(formatter)
    root_logger = logging.getLogger('')
    root_logger.addHandler(console_handler)

    # Avoid logs when sending http requests:
    urllib_logger = logging.getLogger('urllib3.connectionpool')
    urllib_logger.setLevel(logging.ERROR)
    # Remove ui and communication layer logs:
    comm_logger = logging.getLogger('infrastructure.communication')
    comm_logger.setLevel(logging.ERROR)
    ui_logger = logging.getLogger('pydcop.agent.ui')
    ui_logger.setLevel(logging.ERROR)

    levels = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG
    }

    root_logger.setLevel(levels[level])
    console_handler.setLevel(levels[level])
    console_handler.setFormatter(formatter)

    if level == 1:
        logging.warning('logging: warning')
    elif level == 2:
        logging.info('logging: info')
    elif level == 3:
        logging.debug('logging: debug')


if __name__ == '__main__':
    main()
