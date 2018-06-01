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


#make sure it's working

'''
add_test('ops', 'matmul1,157', 'i,a,1,157;i,b,157,19;o,y,1,19;matmul,a,b,y')
add_test('ops', 'matmul307,1', 'i,a,307,1;i,b,1,49;o,y,307,49;matmul,a,b,y')
add_test('ops', 'matmul1,500', 'i,a,1,500;i,b,500,1;o,y,1,1;matmul,a,b,y')
add_test('ops', 'matmul4', 'i,a,4,4;i,b,4,4;o,y,4,4;matmul,a,b,y')
add_test('ops', 'matmul64', 'i,a,64,64;i,b,64,64;o,y,64,64;matmul,a,b,y')
add_test('ops', 'matmul256', 'i,a,256,256;i,b,256,256;o,y,256,256;matmul,a,b,y')
add_test('ops', 'matmul141,73', 'i,a,141,73;i,b,73,89;o,y,141,89;matmul,a,b,y')
add_test('ops', 'matmul21,79', 'i,a,21,79;i,b,79,45;o,y,21,45;matmul,a,b,y')
add_test('ops', 'matmul79,21', 'i,a,79,21;i,b,21,109;o,y,79,109;matmul,a,b,y')
add_test('ops', 'matmul143,79', 'i,a,143,79;i,b,79,143;o,y,143,143;matmul,a,b,y')
'''

#speed tests
add_test('ops', 'matmul512', 'i,a,512,512;i,b,512,512;o,y,512,512;matmul,a,b,y')
add_test('ops', 'matmul1024', 'i,a,1024,1024;i,b,1024,1024;o,y,1024,1024;matmul,a,b,y')
add_test('ops', 'matmul1417,733', 'i,a,1417,733;i,b,733,893;o,y,1417,893;matmul,a,b,y')
add_test('ops', 'matmul733,1417', 'i,a,733,1417;i,b,1417,733;o,y,733,733;matmul,a,b,y')

'''
add_test('ops', 'vadd1', 'i,a,14;i,b,14;o,y,14;vadd,a,b,y')
add_test('ops', 'vadd2', 'i,a,14,7;i,b,14,7;o,y,14,7;vadd,a,b,y')
add_test('ops', 'vadd3', 'i,a,14,7,9;i,b,14,7,9;o,y,14,7,9;vadd,a,b,y')

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

add_test('ops', 'softmax6,4', 'i,x,6,4;o,y,6,4;softmax,x,y')
add_test('ops', 'softmax6,6', 'i,x,6,6;o,y,6,6;softmax,x,y')
add_test('ops', 'softmax6,8', 'i,x,6,8;o,y,6,8;softmax,x,y')
add_test('ops', 'softmax1,12347', 'i,x,1,12347;o,y,1,12347;softmax,x,y')
add_test('ops', 'softmax12347,1', 'i,x,12347,1;o,y,12347,1;softmax,x,y')
add_test('ops', 'softmax327,539', 'i,x,327,539;o,y,327,539;softmax,x,y')
add_test('ops', 'softmax812,113', 'i,x,812,113;o,y,812,113;softmax,x,y')


add_test('ops', 'log_softmax6,4', 'i,x,6,4;o,y,6,4;log_softmax,x,y')
add_test('ops', 'log_softmax6,6', 'i,x,6,6;o,y,6,6;log_softmax,x,y')
add_test('ops', 'log_softmax6,8', 'i,x,6,8;o,y,6,8;log_softmax,x,y')
add_test('ops', 'log_softmax1,12347', 'i,x,1,12347;o,y,1,12347;log_softmax,x,y')
add_test('ops', 'log_softmax12347,1', 'i,x,12347,1;o,y,12347,1;log_softmax,x,y')
add_test('ops', 'log_softmax327,539', 'i,x,327,539;o,y,327,539;log_softmax,x,y')
add_test('ops', 'log_softmax812,113', 'i,x,812,113;o,y,812,113;log_softmax,x,y')


add_test('ops', 'softmax_lcost6,4', 'i,a,6,4;i,b,6,4;o,y,1;softmax_lcost,a,b,y')
add_test('ops', 'softmax_lcost6,6', 'i,a,6,6;i,b,6,6;o,y,1;softmax_lcost,a,b,y')
add_test('ops', 'softmax_lcost6,8', 'i,a,6,8;i,b,6,8;o,y,1;softmax_lcost,a,b,y')
add_test('ops', 'softmax_lcost1,12347', 'i,a,1,12347;i,b,1,12347;o,y,1;softmax_lcost,a,b,y')
add_test('ops', 'softmax_lcost12347,1', 'i,a,12347,1;i,b,12347,1;o,y,1;softmax_lcost,a,b,y')
add_test('ops', 'softmax_lcost327,539', 'i,a,327,539;i,b,327,539;o,y,1;softmax_lcost,a,b,y')
add_test('ops', 'softmax_lcost812,113', 'i,a,812,113;i,b,812,113;o,y,1;softmax_lcost,a,b,y')
'''

ts = json_ts_reader.JsonTsReader(builder.tests, True).ts
if not os.path.isfile(ERRORS_PATH):
    ts.set_run_init(True)
ts.out_err = open(ERRORS_PATH, 'w')
ts.run()
