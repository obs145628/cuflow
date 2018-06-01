import sys
import tensors_saver

import numpy as np
import tensorflow as tf


def run(cmdp, inp, outp):
    model = Model(cmdp, inp, outp)
    model.run()

class Model:

    def __init__(self, cmdp, inp, outp):
        self.cmdp = cmdp
        self.inp = inp
        self.outp = outp
        
        self.need_init = False
        self.inputs = dict()
        self.outputs = dict()
        self.inodes = dict()
        self.sess = tf.Session()

        self.input_list = []
        self.output_list = []

    def run(self):
        with open(self.cmdp, 'r') as f:
            for l in  f.readlines():
                args = l.strip().split(',')
                args = [x.strip() for x in args]
                getattr(self, 'cmd_' + args[0])(args)


        in_saver = tensors_saver.Saver(self.inp)
        for x in self.input_list: in_saver.add(self.inputs[x])
        in_saver.save()
        
        out_saver = tensors_saver.Saver(self.outp)
        for x in self.output_list: out_saver.add(self.outputs[x])
        out_saver.save()

    def add_input(self, name, val):
        val = val.astype(np.float32)
        self.inputs[name] = val
        self.inodes[name] = tf.Variable(val, tf.float32)
        self.need_init = True
        self.input_list.append(name)

    def add_output(self, name, val):
        self.outputs[name] = val
        self.output_list.append(name)
        

    def input(self, name):
        return self.inodes[name]

    def set_output(self, name, node):
        if self.need_init:
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.need_init = False

        val = self.sess.run(node).astype(np.float32)
        if val.shape != self.outputs[name].shape:
            raise Exception('Invalid output shape')
        self.outputs[name] = val








    def cmd_i(self, args):
        name = args[1]
        dims = [int(x) for x in args[2:]]
        x = np.random.random_sample(dims)
        self.add_input(name, x)
        print('Create input {} with shape {}'.format(name, x.shape))

    def cmd_o(self, args):
        name = args[1]
        dims = [int(x) for x in args[2:]]
        x = np.zeros(dims)
        self.add_output(name, x)
        print('Create output {} with shape {}'.format(name, x.shape))

    def cmd_vadd(self, args):
        print('Execute tf.add')
        res = tf.add(self.input(args[1]), self.input(args[2]))
        self.set_output(args[3], res)

    def cmd_log_softmax(self, args):
        print('Execute tf.nn.log_softmax')
        res = tf.nn.log_softmax(self.input(args[1]))
        self.set_output(args[2], res)

    def cmd_matmul(self, args):
        print('Execute tf.matmul')
        res = tf.matmul(self.input(args[1]), self.input(args[2]))
        self.set_output(args[3], res)

    def cmd_softmax(self, args):
        print('Execute tf.nn.softmax')
        res = tf.nn.softmax(self.input(args[1]))
        self.set_output(args[2], res)

    def cmd_softmax_lcost(self, args):
        print('Execute tf.nn.softmax_cross_entropy_with_logits')
        res = tf.nn.softmax_cross_entropy_with_logits(labels=self.input(args[1]),
                                                      logits=self.input(args[2]))
        res = tf.reduce_mean(res)
        res = tf.reshape(res, [1])
        
        self.set_output(args[3], res)

    def cmd_sum(self, args):
        print('Execute tf.reduce_sum')
        res = tf.reduce_sum(self.input(args[1]))
        res = tf.reshape(res, [1])
        self.set_output(args[2], res)
