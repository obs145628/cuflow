import os
import sys
import model


def exec_fail(cmd):
    sys.stdout.flush()
    res = os.system(cmd) >> 8
    if res != 0:
        print('=========== THE TEST FAILED ==========')
        sys.exit(res)

cmd = sys.argv[1]


CMDP = '/tmp/cmd.txt'


INP = sys.argv[2] if len(sys.argv) > 4 else '/tmp/in.npz' 
OUTP = sys.argv[3] if len(sys.argv) > 4 else '/tmp/out.npz'
OUTP2 = sys.argv[4] if len(sys.argv) > 4 else '/tmp/out.tbin'
ROOT_DIR = sys.argv[5] if len(sys.argv) > 6 else '..'
BUILD_DIR = sys.argv[6] if len(sys.argv) > 6 else '.'

TOCHA_DIR = os.path.join(ROOT_DIR,  'ext/tocha/_build/bin')
TBIN_DUMP = os.path.join(TOCHA_DIR, 'tbin-dump')
TBIN_DIFF = os.path.join(TOCHA_DIR, 'tbin-diff')
CUFLOW = os.path.join(BUILD_DIR, 'cuflow')

with open(CMDP, 'w') as f:
    f.write(cmd.replace(';', '\n'))

print('Run model with python:')
model.run(CMDP, INP, OUTP)
print()

print('Run model with cuflow:')
exec_fail('{} {} {} {}'.format(CUFLOW, CMDP, INP, OUTP2))
print()

print('Input: ')
exec_fail('{} {}'.format(TBIN_DUMP, INP))

print('Output (PYTHON): ')
exec_fail('{} {}'.format(TBIN_DUMP, OUTP))

print('Output (CUFLOW): ')
exec_fail('{} {}'.format(TBIN_DUMP, OUTP2))

print('DIFF: ')
exec_fail('{} {} {}'.format(TBIN_DIFF, OUTP, OUTP2))
