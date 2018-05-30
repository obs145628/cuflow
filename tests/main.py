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

def add_test(cat, sub, cmd, mode = None):

    if mode is None:
        add_test(cat, sub, cmd, 'CPU')
        add_test(cat, sub, cmd, 'GPU')
        return

    cat = mode.lower() + '_' + cat

    builder.add_test(cat, sub,
                cmd = [
                    'python', CHECK_BIN, cmd,
                    os.path.join(TEST_BUILD_DIR, 'test_' + cat + '_' + sub + '_in.npz'),
                    os.path.join(TEST_BUILD_DIR, 'test_' + cat + '_' + sub + '_out.npz'),
                    os.path.join(TEST_BUILD_DIR, 'test_' + cat + '_' + sub + '_out.tbin'),
                    ROOT_DIR,
                    BUILD_DIR
                ],
                env = {
                    'ARCH_MODE': mode
                },
                code = 0)

'''
add_test('ops', 'vadd1', 'i,a,14;i,b,14;o,y,14;vadd,a,b,y')
add_test('ops', 'vadd2', 'i,a,14,7;i,b,14,7;o,y,14,7;vadd,a,b,y')
add_test('ops', 'vadd3', 'i,a,14,7,9;i,b,14,7,9;o,y,14,7,9;vadd,a,b,y')
'''

'''
add_test('ops', 'matmul1', 'i,a,141,73;i,b,73,89;o,y,141,89;matmul,a,b,y')
add_test('ops', 'matmul1', 'i,a,1417,733;i,b,733,893;o,y,1417,893;matmul,a,b,y')
'''

'''
add_test('ops', 'sum1', 'i,x,1;o,y,1;sum,x,y')
add_test('ops', 'sum2', 'i,x,2;o,y,1;sum,x,y')
add_test('ops', 'sum3', 'i,x,3;o,y,1;sum,x,y')
add_test('ops', 'sum12', 'i,x,12;o,y,1;sum,x,y')
add_test('ops', 'sum67', 'i,x,67;o,y,1;sum,x,y')
add_test('ops', 'sum129', 'i,x,129;o,y,1;sum,x,y')
add_test('ops', 'sum256', 'i,x,256;o,y,1;sum,x,y')
add_test('ops', 'sum2456', 'i,x,2456;o,y,1;sum,x,y')
add_test('ops', 'sum12357', 'i,x,12357;o,y,1;sum,x,y')
add_test('ops', 'sum119357', 'i,x,119357;o,y,1;sum,x,y')
add_test('ops', 'sum40M', 'i,x,40193577;o,y,1;sum,x,y')
'''

add_test('ops', 'softmax6,4', 'i,x,6,4;o,y,6,4;softmax,x,y')
add_test('ops', 'softmax6,6', 'i,x,6,6;o,y,6,6;softmax,x,y')
add_test('ops', 'softmax6,8', 'i,x,6,8;o,y,6,8;softmax,x,y')
add_test('ops', 'softmax1,12347', 'i,x,1,12347;o,y,1,12347;softmax,x,y')
add_test('ops', 'softmax12347,1', 'i,x,12347,1;o,y,12347,1;softmax,x,y')
add_test('ops', 'softmax327,539', 'i,x,327,539;o,y,327,539;softmax,x,y')
add_test('ops', 'softmax812,113', 'i,x,812,113;o,y,812,113;softmax,x,y')

ts = json_ts_reader.JsonTsReader(builder.tests, True).ts
if not os.path.isfile(ERRORS_PATH):
    ts.set_run_init(True)
ts.out_err = open(ERRORS_PATH, 'w')
ts.run()
