import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

from M_Config import *

from Trans_Encoder_No_Emb import *


class Layer_Attend( Layer ) :
     
    def __init__( self, queries, dim_model, **kwargs ) :
        super().__init__( **kwargs )  # Do NOT put 'dynamic = True' here. It leads to 'NotImplementedError'. DO NOT put trainable = ...

        self.queries = queries
        self.dim_model = dim_model

        self.encoder = Encoder( num_layers = 6, d_model = dim_model, num_heads = int( dim_model / 4 ), dff = int( dim_model * 2 ), rate = 0.1 )

    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'queries': self.queries,
        })
        return config

    def build( self, input_shape ) : # input_shape = ( None, self.nQueries )

        #self.thresholds = self.add_weight(name = 'thresholds', shape = ( input_shape[-1], ), \
        #    initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.), trainable = True, dtype = tf.float32 )
        #self.maxBetUnit = self.add_weight(name = 'maxBetUnit', shape = ( self.nQueries, 1 ), initializer = 'uniform', trainable = True, dtype = tf.float32 )

        super().build(input_shape)
        return


    def call( self, inputs, training ) :

        x = inputs

        x = self.encoder( x, training = training, mask = None )

        return x
    
    
    def compute_output_shape( self, input_shape ) :
         output_shape = input_shape

         return output_shape
    