import tensorflow as tf
from keras.layers import Layer
from keras import backend as K
import numpy as np
from QSM_func import *
from model_structures import *
  

class CustGradClass_br:

    def __init__(self,x1,x2,x3,smv,trun,opt):
        self.f = tf.custom_gradient(lambda x: CustGradClass_br._f(self, x))
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.smv= smv
        self.trun=trun
        self.opt = opt
        
    @staticmethod
    def _f(self, x):
        fx = conjgrad_br_trun(self.x1, self.x2, self.x3, x, self.smv, self.trun, self.opt)
        def grad(dy):
            grad = conjgrad_br_trun_grad(self.x1, self.x2, self.x3, dy, self.smv, self.trun, self.opt) # compute gradient
            return grad
        return fx, grad


class CustomLayer_br(Layer):
    def __init__(self, init_x, x, smv, trun, opt):

        self.c = CustGradClass_br(x[...,0:len(opt['rad'])],init_x,x[...,len(opt['rad']):len(opt['rad'])+1], smv, trun, opt)
        super(CustomLayer_br, self).__init__()

    def call(self, inputdata):
        
        return self.c.f(inputdata)

