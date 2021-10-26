import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

from M_Config import *

class Layer_Softmax( Layer ) :
     
    def __init__( self, queries, **kwargs ) :
        super().__init__( **kwargs ) # Do NOT put 'dynamic = True' here. It leads to 'NotImplementedError'. DO NOT put trainable = ...

        self.queries = queries
        self.SoftMax = tf.keras.layers.Softmax( axis = -1 )

        #self.threshold = self.add_weight(name = 'thresholds', shape = ( 3, ), \
        #    initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.), trainable = True, dtype = tf.float32 )

        inputs = tf.keras.layers.Input( shape = (3,), dtype = tf.float32 )
        x = tf.keras.layers.BatchNormalization() ( inputs )
        x = tf.keras.layers.Dense( units = 12, activation = 'sigmoid' )( x )
        # x = tf.keras.layers.BatchNormalization() ( x )
        x = tf.keras.layers.Dense( units = 12, activation = 'relu' )( x )
        x = tf.keras.layers.BatchNormalization() ( x )
        x = tf.keras.layers.Dense( units = 3, activation = 'sigmoid' )( x )
        #x = x * tf.math.sigmoid( ( x - self.threshold ) )
        self.model_zero_out_small = tf.keras.Model( inputs = inputs, outputs = x )

        return

    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'queries': self.queries
        })
        return config

    def build( self, input_shape ) : # input_shape = ( None, self.nQueries )
      
        super().build(input_shape)
      
        return 

    def call( self, inputs, training ) : # inputs.shape = ( None, self.nQueries )

        bp = tf.split( inputs, inputs.shape[-1], axis = -1 ) # bp[k].shape = (None, 1)
        bp_list = []

        # Be careful: queries might be inconsistent.

        #if bp.count(0) * bp.count(1) * bp.count(2) > 0 : # "Home__Win", "Game_Draw", and "Away__Win" queries are found.
        bp_list += self.Get_Softmax( [ bp[0], bp[1], bp[2] ] )
        #if bp.count(3) * bp.count(4) > 0 : # "Both_Sc_Y" and "Both_Sc_N" queries are found.
        bp_list += self.Get_Softmax( [ bp[3], bp[4] ] )
        #if bp.count(5) * bp.count(6) > 0 : # "Over__1.5" and "Under_1.5" queries are found.
        bp_list += self.Get_Softmax( [ bp[5], bp[6] ] ) #================================================= Work?
        #if bp.count(7) * bp.count(8) > 0 : # "Over__2.5" and "Under_2.5" queries are found.
        bp_list += self.Get_Softmax( [ bp[7], bp[8] ] )

        # "Score_1_0", "Score_2_1"and "Score_1_1" queries don't make a complet one, actually.
        #if bp.count(9) * bp.count(10) * bp.count(11) > 0 : # "Score_1_0", "Score_2_1"and "Score_1_1" queries are found.
        bp_list += self.Get_Softmax( [ bp[9], bp[10], bp[11] ] ) # Do NOT softmax, as the might not, in most cases, sum up to 1.

        ret = tf.concat( bp_list, axis = -1 )

        return ret
    

    def compute_output_shape( self, input_shape ) :
        output_shape = input_shape

        return output_shape


    def Get_Softmax( self, bp_list ) :
        bp_tensor = tf.concat( bp_list, axis = -1 ) # bp_tensor.shape = (None, N)
        bp_tensor = self.SoftMax( bp_tensor )
        bp_list_return = tf.split( bp_tensor, bp_tensor.shape[-1], axis = -1 )

        return  bp_list_return
    
    
    def Zero_out_Small( self, bp_list ) :
        bp_tensor = tf.concat( bp_list, axis = -1 ) # bp_tensor.shape = (None, N)
        bp_tensor = self.model_zero_out_small( bp_tensor )
        bp_list_return = tf.split( bp_tensor, bp_tensor.shape[-1], axis = -1 )

        return  bp_list_return
