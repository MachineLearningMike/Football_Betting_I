import tensorflow as tf
import numpy as np

from LossByQuery import *

class FilterLayer(tf.keras.layers.Layer):
   
    def __init__(self, filterFunc, name, **kwargs):
        super(FilterLayer, self).__init__(name = name, trainable = False, dynamic = True, **kwargs)

        self.filterFunc = filterFunc

    def build(self, input_shape):
        super(FilterLayer, self).build(input_shape)

    def call(self, inputs, training = False):
        ret = self.filterFunc( inputs[:, 0], inputs[:, 1] )
        #ret = np.expand_dims( ret, axis = -1)  # no gradient
        #ret = tf.Variable( ret )               # no gradient
        return ret

    def compute_output_shape(self, input_shape):
        tf.print( '==================FilterLayer input shape ', input_shape)
        temp = list( input_shape )
        del temp[-1]
        #temp[-1] = 1
        output_shape = tf.TensorShape( temp )
        return output_shape


class SqueezeSequenceEmbedding(tf.keras.layers.Layer):

    useLargeBert = False
   
    def __init__(self,  **kwargs):
        super(SqueezeSequenceEmbedding, self).__init__(name = name, trainable = False, dynamic = True, **kwargs)

    def build(self, input_shape):
        super(SqueezeSequenceEmbedding, self).build(input_shape)

    def call(self, inputs, training = False):
        inputs = tf.keras.backend.squeeze( intuts, axist = -2 )
        return ret

    def compute_output_shape(self, input_shape):
        temp = list( input_shape )
        del temp[-2]
        output_shape = tuple( temp )
        return output_shape



class Linear(tf.keras.layers.Layer):
  def __init__(self, name, units=32, input_dim=32 ):
    super(Linear, self).__init__( name = name )
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                              dtype='float32'),
                         trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                              dtype='float32'),
                         trainable=True)

  def call(self, inputs):
    ret = tf.matmul(inputs, self.w) + self.b
    ret = inputs[:, 0] + inputs[:, 1]
    ret = tf.Variable(ret)
    #ret = Metrics_Sigmoid_By_Query[0] ( [ inputs[:, 0] ], [ inputs[:, 1] ] )
    return ret
