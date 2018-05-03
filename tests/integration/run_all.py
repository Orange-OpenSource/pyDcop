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



import os
import time


def run_all_integ_tests():
    """ Run all integration tests in the directory """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    tests_run = {}
    tests_nums = [0, 0, 0]

    entries = os.listdir(dir_path)
    for entry in entries:
        module_name = entry[:-3]
        if entry.endswith('.py'):
            integ_test_module = __import__(entry[:-3])

            try:
                test_method = getattr(integ_test_module, 'run_test')
                tests_nums[0] += 1
                try:
                    print('Running test from ' + module_name)
                    result = test_method()
                    if result == 0:
                        tests_run[module_name] = 'OK'
                        tests_nums[1] += 1
                    else:
                        tests_run[module_name] = 'Fail'
                        tests_nums[2] += 1

                except Exception as e:
                    tests_run[module_name] = 'ERROR : {}'.format(e)
                    tests_nums[2] += 1
            except AttributeError:
                print('No "run_test" method in '+ entry)
                tests_run[module_name] = 'No tests to run'

    time.sleep(1)

    # Print summary of runs :
    print('\n\n   INTEG TEST SUMMARY : {} tests, {} success, {} error '.format(
        *tests_nums))

    print('FAILED TESTS:')
    for k in [k  for k,v in tests_run.items() if v=='Fail'] :
        print(' * ' + k)

    print('TESTS WITH ERROR:')
    for (k, v) in [(k, v) for k, v in tests_run.items() if v.startswith(\
            'ERROR')]:
        print('* {} : {}'.format(k, v))

    print('FILES WITH NO TEST:')
    for k in [k for k, v in tests_run.items() if v.startswith( 'No tests')]:
        print(' * ' + k)

if __name__ == '__main__':
    run_all_integ_tests()