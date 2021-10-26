import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

from M_Config import *


class Layer_Stake( tf.keras.layers.Layer ) :
     
    def __init__( self, queries, max_alpha, max_beta, **kwargs ) :
        super().__init__( **kwargs )  # Do NOT put 'dynamic = True' here. It leads to 'NotImplementedError'. DO NOT put trainable = ...

        self.queries = queries
        self.max_alpha = max_alpha
        self.max_beta = max_beta

    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'queries': self.queries,
            'max_alpha' : self.max_alpha,
            'max_beta' : self.max_beta,
        })

        return config

    def build( self, input_shape ) : # input_shape = ( None, len( self.queries ) )

        self.alpha = self.add_weight( 'alpha', shape = ( input_shape[-1], ) , \
            initializer = tf.keras.initializers.RandomUniform(minval = 0., maxval = self.max_alpha), \
            constraint = tf.keras.constraints.MinMaxNorm( min_value = 0.0, max_value = tf.sqrt( tf.math.square(self.max_alpha) * len(self.queries) ) ), \
            trainable = True, dtype = tf.float32 )
        
        self.beta = self.add_weight( 'beta', shape = ( input_shape[-1], ), \
            initializer = tf.keras.initializers.RandomUniform(minval = 0., maxval = self.max_beta), \
            constraint = tf.keras.constraints.MinMaxNorm( min_value = 1.0, max_value = tf.sqrt( tf.math.square(self.max_beta) * len(self.queries) ) ), \
            trainable = True, dtype = tf.float32 )

        super().build(input_shape)

        return


    def call( self, inputs, training ) :

        bet_worth = inputs
        bet_stake = tf.add( tf.multiply( bet_worth, self.alpha ), self.beta  )
        bet_stake = bet_stake

        return bet_stake
    
    
    def compute_output_shape( self, input_shape ) :
         output_shape = input_shape

         return output_shape

"""
x = tf.Variable( [ [1, 2], [2, 3], [3, 4] ], dtype = tf.float32  )
print( 'x = ', x )
stake = Layer_Stake( [1,2,3], 1.0, 2.0 )
y = stake( x )
print( 'y = ', y )
print( 'alpha, beta = ', stake.alpha, stake.beta )
"""