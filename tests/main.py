import os
import sys

import json_ts_builder
import json_ts_reader
import tensors_saver


ROOT_DIR = sys.argv[1]
BUILD_DIR = sys.argv[2]
TEST_BUILD_DIR = BUILD_DIR
TEST_DIR = os.path.join(ROOT_DIR, 'tests/')
ERRORS_PATH = os.path.join(BUILD_DIR, 'errors.log')
CHECK_BIN = os.path.join(TEST_DIR, 'check.py')


builder = json_ts_builder.JsonTsBuilder()

def add_test(cat, sub, cmd):

    builder.add_test(cat, sub,
                cmd = [
                    'python', CHECK_BIN, cmd,
                    os.path.join(TEST_BUILD_DIR, 'test_' + cat + '_' + sub + '_in.npz'),
                    os.path.join(TEST_BUILD_DIR, 'test_' + cat + '_' + sub + '_out.npz'),
                    os.path.join(TEST_BUILD_DIR, 'test_' + cat + '_' + sub + '_out.tbin'),
                    ROOT_DIR,
                    BUILD_DIR
                ],
                code = 0)


add_test('ops', 'vadd1', 'i,a,14;i,b,14;o,y,14;vadd,a,b,y')

ts = json_ts_reader.JsonTsReader(builder.tests, True).ts
if not os.path.isfile(ERRORS_PATH):
    ts.set_run_init(True)
ts.out_err = open(ERRORS_PATH, 'w')
ts.run()
