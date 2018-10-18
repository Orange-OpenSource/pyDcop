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
.. _pydcop_commands_consolidate:

pydcop consolidate
==================

``pydcop consolidate`` extract main statistics from a result file.

Synopsis
--------

::

  pydcop consolidate <result_file>


Description
-----------

extracting
 extract end metrics from a json result output file.
 produces a csv line, which can be appended to a file

sampling
 takes a run time metrics files, produced with the 'value_change' mode, and
 generate a sample run time metric file


average
 takes a csv file

"""
import logging
import json
import csv
from typing import List

logger = logging.getLogger("pydcop.cli.consolidate")


def set_parser(subparsers):
    parser = subparsers.add_parser(
        "consolidate", help="Various utilities to consolidate data"
    )
    parser.set_defaults(func=run_cmd)

    parser.add_argument("files", type=str, nargs="+", help="file(s)")

    parser.add_argument(
        "--extract",
        action="store_true",
        default=False,
        help="Extract result from a json output file",
    )


def run_cmd(args):

    if args.extract:
        files = args.files
        if args.output:
            # TODO: init file with headers
            with open(args.output, mode="a", newline="") as output_file:
                extract(files, output_file)
        else:
            target = extract(files, WriterTarget())
            print(target.writen)


def extract(files: List[str], target):

    for file in files:
        logger.debug(f"analyse file {file}")
        with open(file, mode="r") as f:

            data_json = json.load(f)
            data = [
                data_json["time"],
                data_json["cost"],
                data_json["cycle"],
                data_json["msg_count"],
                data_json["msg_size"],
                data_json["status"],
            ]
            csv_writer = csv.writer(target)
            csv_writer.writerow(data)

    return target


class WriterTarget:

    writen: str = ""

    def write(self, s):
        self.writen += s
